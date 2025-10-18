package data

import (
	"fmt"
	"strings"
)

// 统一入口：根据数据集类型选择加载器
// 数据集类型："MNIST" | "FASHION"
func NewLoader(dataset, root string, bs int, seed int64) (*Loader, error) {
	name := strings.TrimSpace(dataset)
	if name == "" {
		name = "MNIST"
	}
	switch strings.ToUpper(name) {
	case "MNIST", "FASHION":
		// FASHION 与 MNIST 同 IDX 格式，文件名通常亦相同；根目录指向对应路径即可
		return NewLoaderMNIST(root, bs, seed)
	default:
		return nil, fmt.Errorf("unsupported dataset: %s", name)
	}
}
