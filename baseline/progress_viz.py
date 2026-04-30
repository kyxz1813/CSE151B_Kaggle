import time
import pandas as pd
import matplotlib.pyplot as plt


class RunProgressDashboard:
    def __init__(
        self,
        run_name="run",
        total=None,
        score_available=True,
        render_every=1,
        show_accuracy=True,
        show_timing=True,
        max_recent_rows=5,
    ):
        self.run_name = run_name
        self.total = total
        self.score_available = score_available
        self.render_every = render_every
        self.show_accuracy = show_accuracy and score_available
        self.show_timing = show_timing
        self.max_recent_rows = max_recent_rows
        self.started_at = time.perf_counter()
        self.rows = []
        self.last_render_step = 0

    def update(self, row, force=False):
        now = time.perf_counter()
        step = len(self.rows) + 1
        elapsed = now - self.started_at

        enriched = dict(row)
        enriched["step"] = step
        enriched["elapsed_sec"] = elapsed
        enriched["sec_per_problem"] = elapsed / step if step else None

        if self.total:
            remaining = max(self.total - step, 0)
            enriched["eta_sec"] = remaining * enriched["sec_per_problem"]
            enriched["progress_frac"] = step / self.total
        else:
            enriched["eta_sec"] = None
            enriched["progress_frac"] = None

        if self.score_available:
            scored = [r for r in self.rows + [enriched] if r.get("correct") is not None]
            if scored:
                enriched["running_accuracy"] = sum(bool(r.get("correct")) for r in scored) / len(scored)
            else:
                enriched["running_accuracy"] = None
        else:
            enriched["running_accuracy"] = None

        self.rows.append(enriched)

        should_render = force or step == self.total or step - self.last_render_step >= self.render_every
        if should_render:
            self.render()
            self.last_render_step = step

    def dataframe(self):
        return pd.DataFrame(self.rows)

    def summary(self):
        df = self.dataframe()
        out = {
            "run_name": self.run_name,
            "n": len(df),
            "total": self.total,
        }

        if len(df):
            out["elapsed_sec"] = float(df["elapsed_sec"].iloc[-1])
            out["sec_per_problem"] = float(df["sec_per_problem"].iloc[-1])
            if self.total:
                out["eta_sec"] = float(df["eta_sec"].iloc[-1])

        if self.score_available and len(df) and "correct" in df.columns:
            scored = df[df["correct"].notna()]
            if len(scored):
                out["accuracy"] = float(scored["correct"].astype(bool).mean())
                out["n_correct"] = int(scored["correct"].astype(bool).sum())
                out["n_scored"] = int(len(scored))

        return out

    def render(self):
        try:
            from IPython.display import clear_output, display
            clear_output(wait=True)
        except Exception:
            clear_output = None
            display = None

        summary = self.summary()
        print("=" * 80)
        print(self.run_name)
        print("=" * 80)
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")

        df = self.dataframe()

        if display is not None and len(df):
            cols = [c for c in ["step", "id", "is_mcq", "correct", "elapsed_sec", "sec_per_problem", "eta_sec", "running_accuracy"] if c in df.columns]
            display(df[cols].tail(self.max_recent_rows))

        if len(df) < 2:
            return

        try:

            if self.show_accuracy and "running_accuracy" in df.columns:
                acc_df = df[df["running_accuracy"].notna()]
                if len(acc_df):
                    plt.figure(figsize=(7, 3))
                    plt.plot(acc_df["step"], acc_df["running_accuracy"])
                    plt.xlabel("Problems completed")
                    plt.ylabel("Running accuracy")
                    plt.title(f"{self.run_name}: accuracy")
                    plt.ylim(0, 1)
                    plt.show()

            if self.show_timing and "sec_per_problem" in df.columns:
                plt.figure(figsize=(7, 3))
                plt.plot(df["step"], df["sec_per_problem"])
                plt.xlabel("Problems completed")
                plt.ylabel("Seconds / problem")
                plt.title(f"{self.run_name}: speed")
                plt.show()

        except Exception as exc:
            print("Progress plots unavailable:", exc)

    def finish(self):
        self.render()
        return self.summary()