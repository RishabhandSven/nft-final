"""Run wash trading pipeline with 1.5M sample."""
from pathlib import Path
from ai_engine.wash_trading_pipeline import WashTradingTrainer, DATA_PATH

trainer = WashTradingTrainer(
    data_path=DATA_PATH,
    target_sample_size=1_500_000,
    contamination=0.05,
    n_estimators=100,
    n_jobs=6,  # Zenbook-friendly
)
trainer.run(output_dir=Path(__file__).parent / "data" / "results")
