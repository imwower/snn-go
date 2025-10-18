package data

// 统一入口：根据数据集类型选择加载器
// 数据集类型："MNIST" | "FASHION" | "SYNTH"
func NewLoader(dataset, root string, bs int, seed int64) (*Loader, error) {
	switch dataset {
	case "MNIST", "FASHION":
		// FASHION 与 MNIST 同 IDX 格式，文件名通常亦相同；根目录指向对应路径即可
		return NewLoaderMNIST(root, bs, seed)
	case "SYNTH":
		return synth(bs, seed), nil
	default:
		// 默认尝试 MNIST，失败则回退合成
		if ld, err := NewLoaderMNIST(root, bs, seed); err == nil {
			return ld, nil
		}
		return synth(bs, seed), nil
	}
}
