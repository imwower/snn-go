package main

import (
	"context"
	"log"

	"github.com/imwower/snn-go/internal/config"
	"github.com/imwower/snn-go/internal/trainer"
)

func main() {
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)
	log.SetPrefix("[TRAIN] ")

	cfg, err := config.Load("config.yaml")
	if err != nil {
		log.Fatalf("加载配置失败：%v", err)
	}
	log.Printf("配置：dataset=%s epochs=%d batch=%d T=%d K=%d lr=%.4f",
		cfg.Training.Dataset, cfg.Training.Epochs, cfg.Training.BatchSize,
		cfg.Training.Timesteps, cfg.Training.FixedPointK, cfg.Training.LR,
	)
	runner := trainer.NewRunner(cfg, func(level, msg string) {
		log.Printf("[%s] %s", level, msg)
	}, func(status string) {
		log.Printf("[TRAIN] 状态变更：%s", status)
	})

	opts := trainer.Options{
		Dataset:       cfg.Training.Dataset,
		DataRoot:      cfg.Training.DataRoot,
		Epochs:        cfg.Training.Epochs,
		BatchSize:     cfg.Training.BatchSize,
		Timesteps:     cfg.Training.Timesteps,
		FixedPointK:   cfg.Training.FixedPointK,
		FixedPointTol: cfg.Training.FixedPointTol,
		LearningRate:  cfg.Training.LR,
		Hidden:        cfg.Training.Hidden,
		Mode:          "tstep",
		Layers:        1,
		NetworkSize:   cfg.Training.Hidden,
		Seed:          cfg.Training.Seed,
		EndToEnd:      cfg.Training.EndToEnd,
		InputSize:     cfg.Model.Input,
		OutputSize:    cfg.Model.Output,
		KappaBS:       cfg.Training.KappaBS,
		KappaAS:       cfg.Training.KappaAS,
		Theta:         cfg.Training.Theta,
		AlphaB:        cfg.Training.AlphaB,
		AlphaA:        cfg.Training.AlphaA,
		AlphaS:        cfg.Training.AlphaS,
	}

	if err := runner.Start(opts); err != nil {
		log.Fatalf("启动训练失败：%v", err)
	}
	if err := runner.Wait(context.Background()); err != nil && err != context.Canceled {
		log.Fatalf("训练失败：%v", err)
	}
	log.Println("训练完成，退出")
}
