package datasets

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

type File struct {
	URL  string
	Path string
	Size int64
}

type Definition struct {
	Name   string
	Label  string
	Files  []File
	Folder string
}

type Status struct {
	Name      string  `json:"name"`
	Label     string  `json:"label"`
	Installed bool    `json:"installed"`
	State     string  `json:"state"`
	Progress  float64 `json:"progress"`
	Message   string  `json:"message,omitempty"`
	TimeUnix  int64   `json:"time_unix"`
}

type Manager struct {
	mu        sync.Mutex
	root      string
	base      string
	defs      map[string]Definition
	status    map[string]*Status
	client    *http.Client
	broadcast func(event string, payload []byte)
}

func NewManager(root string, broadcast func(event string, payload []byte)) *Manager {
	if root == "" {
		root = ".data"
	}
	cleanRoot := filepath.Clean(root)
	baseRoot := guessBaseRoot(cleanRoot)
	m := &Manager{
		root:      cleanRoot,
		base:      baseRoot,
		defs:      makeDefinitions(),
		status:    map[string]*Status{},
		client:    &http.Client{Timeout: 30 * time.Second},
		broadcast: broadcast,
	}
	m.refreshAll()
	return m
}

func (m *Manager) datasetRoot(def Definition) string {
	base := m.base
	if base == "" {
		base = m.root
	}
	folder := strings.TrimSpace(def.Folder)
	if folder == "" {
		folder = strings.ToLower(def.Name)
	}
	folder = strings.Trim(folder, `/\`)
	if folder == "" {
		return base
	}
	candidate := filepath.Join(base, folder)
	if samePath(candidate, m.root) {
		return m.root
	}
	return candidate
}

func (m *Manager) DatasetPath(name string) (string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	def, ok := m.defs[strings.ToUpper(name)]
	if !ok {
		return "", ErrUnknownDataset
	}
	return m.datasetRoot(def), nil
}

func (m *Manager) DatasetInstalled(name string) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	status := m.getStatusLocked(name)
	status.Installed = m.isInstalledLocked(name)
	return status.Installed
}

func (m *Manager) List() []Status {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.refreshAllLocked()
	list := make([]Status, 0, len(m.defs))
	for _, def := range m.defs {
		status := m.getStatusLocked(def.Name)
		list = append(list, *status)
	}
	return list
}

func (m *Manager) HandleList(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodOptions {
		setPreflight(w, http.MethodGet)
		return
	}
	setCORS(w, http.MethodGet)
	resp := struct {
		Datasets []Status `json:"datasets"`
	}{
		Datasets: m.List(),
	}
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	_ = json.NewEncoder(w).Encode(resp)
}

type downloadRequest struct {
	Name string `json:"name"`
}

func (m *Manager) HandleDownload(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodOptions {
		setPreflight(w, http.MethodPost)
		return
	}
	setCORS(w, http.MethodPost)
	var req downloadRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}
	name := strings.TrimSpace(req.Name)
	if name == "" {
		http.Error(w, "dataset name required", http.StatusBadRequest)
		return
	}
	log.Printf("数据集：收到下载请求 %s", name)
	if err := m.startDownload(name); err != nil {
		if errors.Is(err, ErrUnknownDataset) {
			log.Printf("数据集：未知数据集 %s", name)
			http.Error(w, err.Error(), http.StatusNotFound)
			return
		}
		if errors.Is(err, ErrDownloadInProgress) {
			w.WriteHeader(http.StatusAccepted)
			_, _ = w.Write([]byte(`{"status":"downloading"}`))
			return
		}
		log.Printf("数据集：启动下载失败 %s：%v", name, err)
		http.Error(w, fmt.Sprintf("download error: %v", err), http.StatusInternalServerError)
		return
	}
	w.WriteHeader(http.StatusAccepted)
	_, _ = w.Write([]byte(`{"status":"started"}`))
}

func setCORS(w http.ResponseWriter, method string) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", method+", OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
}

func setPreflight(w http.ResponseWriter, methods ...string) {
	if len(methods) == 0 {
		methods = []string{http.MethodPost}
	}
	merged := append([]string{http.MethodOptions}, methods...)
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", strings.Join(merged, ", "))
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
	w.WriteHeader(http.StatusNoContent)
}

var (
	ErrUnknownDataset     = errors.New("unknown dataset")
	ErrDownloadInProgress = errors.New("dataset download already in progress")
)

func (m *Manager) startDownload(name string) error {
	m.mu.Lock()
	log.Printf("数据集：startDownload 进入 name=%s", name)
	def, ok := m.defs[strings.ToUpper(name)]
	if !ok {
		m.mu.Unlock()
		log.Printf("数据集：startDownload 未知 name=%s", name)
		return ErrUnknownDataset
	}
	status := m.getStatusLocked(def.Name)
	if status.State == "downloading" {
		m.mu.Unlock()
		log.Printf("数据集：startDownload 已在下载 %s", def.Name)
		return ErrDownloadInProgress
	}
	status.State = "downloading"
	status.Progress = 0
	status.Message = ""
	status.TimeUnix = time.Now().Unix()
	m.mu.Unlock()

	m.emit(def.Name, "start", 0, "")
	log.Printf("数据集：开始异步下载 %s", def.Name)
	go m.download(def)
	return nil
}

func (m *Manager) download(def Definition) {
	root := m.datasetRoot(def)
	log.Printf("数据集：开始下载 %s -> %s", def.Name, root)
	if err := os.MkdirAll(root, 0o755); err != nil {
		m.fail(def.Name, fmt.Errorf("mkdir: %w", err))
		return
	}

	totalExpected := int64(0)
	for _, file := range def.Files {
		totalExpected += file.Size
	}
	if totalExpected == 0 {
		totalExpected = 1
	}
	var downloaded int64

	for _, file := range def.Files {
		dest := filepath.Join(root, file.Path)
		if ok := m.skipIfPresent(dest, file.Size); ok {
			downloaded += file.Size
			m.emitProgress(def.Name, float64(downloaded)/float64(totalExpected), "")
			continue
		}
		if err := m.fetchFile(dest, file, func(delta int64) {
			downloaded += delta
			progress := float64(downloaded) / float64(totalExpected)
			if progress > 1 {
				progress = 1
			}
			m.emitProgress(def.Name, progress, "")
		}); err != nil {
			m.fail(def.Name, err)
			return
		}
	}
	m.succeed(def.Name)
}

func (m *Manager) fetchFile(dest string, file File, onProgress func(delta int64)) error {
	req, err := http.NewRequest(http.MethodGet, file.URL, nil)
	if err != nil {
		return fmt.Errorf("构造请求失败：%w", err)
	}
	resp, err := m.client.Do(req)
	if err != nil {
		return fmt.Errorf("下载失败：%w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("下载失败：状态码 %d", resp.StatusCode)
	}
	tmp := dest + ".part"
	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		return fmt.Errorf("创建目录失败：%w", err)
	}
	out, err := os.Create(tmp)
	if err != nil {
		return fmt.Errorf("写入文件失败：%w", err)
	}
	defer out.Close()

	buf := make([]byte, 64*1024)
	lastEmit := time.Now()
	var copied int64
	for {
		n, readErr := resp.Body.Read(buf)
		if n > 0 {
			if _, err := out.Write(buf[:n]); err != nil {
				return fmt.Errorf("写入文件失败：%w", err)
			}
			copied += int64(n)
			onProgress(int64(n))
			if time.Since(lastEmit) > 250*time.Millisecond {
				lastEmit = time.Now()
				onProgress(0)
			}
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return fmt.Errorf("下载过程中断：%w", readErr)
		}
	}
	if err := out.Sync(); err != nil {
		return fmt.Errorf("同步文件失败：%w", err)
	}
	if err := os.Rename(tmp, dest); err != nil {
		return fmt.Errorf("重命名文件失败：%w", err)
	}
	return nil
}

func (m *Manager) skipIfPresent(dest string, size int64) bool {
	info, err := os.Stat(dest)
	if err != nil {
		return false
	}
	if size > 0 && info.Size() != size {
		return false
	}
	return true
}

func (m *Manager) emit(name, state string, progress float64, message string) {
	m.mu.Lock()
	status := m.getStatusLocked(name)
	if state != "" {
		status.State = state
	}
	if progress >= 0 {
		status.Progress = progress
	}
	status.Message = message
	status.TimeUnix = time.Now().Unix()
	status.Installed = m.isInstalledLocked(name)
	payload, _ := json.Marshal(struct {
		Name      string  `json:"name"`
		State     string  `json:"state"`
		Progress  float64 `json:"progress"`
		Message   string  `json:"message,omitempty"`
		Installed bool    `json:"installed"`
		TimeUnix  int64   `json:"time_unix"`
	}{
		Name:      status.Name,
		State:     status.State,
		Progress:  status.Progress,
		Message:   status.Message,
		Installed: status.Installed,
		TimeUnix:  status.TimeUnix,
	})
	m.mu.Unlock()
	if m.broadcast != nil {
		m.broadcast("dataset_download", payload)
	}
}

func (m *Manager) emitProgress(name string, progress float64, message string) {
	m.emit(name, "progress", progress, message)
}

func (m *Manager) fail(name string, err error) {
	log.Printf("数据集：下载失败 %s：%v", name, err)
	m.emit(name, "error", 0, err.Error())
}

func (m *Manager) succeed(name string) {
	log.Printf("数据集：完成下载 %s", name)
	m.emit(name, "complete", 1, "")
}

func (m *Manager) isInstalledLocked(name string) bool {
	def, ok := m.defs[strings.ToUpper(name)]
	if !ok {
		return false
	}
	root := m.datasetRoot(def)
	for _, file := range def.Files {
		path := filepath.Join(root, file.Path)
		if _, err := os.Stat(path); err != nil {
			return false
		}
	}
	return true
}

func (m *Manager) refreshAll() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.refreshAllLocked()
}

func (m *Manager) refreshAllLocked() {
	for _, def := range m.defs {
		status := m.getStatusLocked(def.Name)
		status.Installed = m.isInstalledLocked(def.Name)
		if !status.Installed && status.State == "" {
			status.State = "idle"
			status.Progress = 0
		}
		if status.Installed {
			status.Progress = 1
		}
	}
}

func (m *Manager) getStatusLocked(name string) *Status {
	if status, ok := m.status[strings.ToUpper(name)]; ok {
		return status
	}
	def := m.defs[strings.ToUpper(name)]
	status := &Status{
		Name:      def.Name,
		Label:     def.Label,
		Installed: false,
		State:     "idle",
		Progress:  0,
		TimeUnix:  time.Now().Unix(),
	}
	status.Installed = m.isInstalledLocked(def.Name)
	if status.Installed {
		status.Progress = 1
	}
	m.status[strings.ToUpper(name)] = status
	return status
}

func makeDefinitions() map[string]Definition {
	return map[string]Definition{
		"MNIST": {
			Name:   "MNIST",
			Label:  "MNIST 手写数字",
			Folder: "mnist",
			Files: []File{
				{URL: "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz", Path: "train-images-idx3-ubyte.gz", Size: 9912422},
				{URL: "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz", Path: "train-labels-idx1-ubyte.gz", Size: 28881},
				{URL: "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz", Path: "t10k-images-idx3-ubyte.gz", Size: 1648877},
				{URL: "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz", Path: "t10k-labels-idx1-ubyte.gz", Size: 4542},
			},
		},
		"FASHION": {
			Name:   "FASHION",
			Label:  "Fashion-MNIST 服饰",
			Folder: "fashion",
			Files: []File{
				{URL: "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz", Path: "train-images-idx3-ubyte.gz", Size: 9912422},
				{URL: "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz", Path: "train-labels-idx1-ubyte.gz", Size: 28881},
				{URL: "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz", Path: "t10k-images-idx3-ubyte.gz", Size: 1648877},
				{URL: "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz", Path: "t10k-labels-idx1-ubyte.gz", Size: 4542},
			},
		},
	}
}

func guessBaseRoot(path string) string {
	cleaned := filepath.Clean(path)
	p := cleaned
	for {
		base := filepath.Base(p)
		if strings.EqualFold(base, ".data") || strings.EqualFold(base, "data") {
			return p
		}
		parent := filepath.Dir(p)
		if parent == "." || parent == "/" || parent == p {
			break
		}
		p = parent
	}
	dir := filepath.Dir(cleaned)
	if dir == "." || dir == "/" {
		return cleaned
	}
	return dir
}

func samePath(a, b string) bool {
	if a == "" || b == "" {
		return false
	}
	return strings.EqualFold(filepath.Clean(a), filepath.Clean(b))
}
