from ranx import Qrels, Run, evaluate
import pandas as pd


def load_qrels(csv_path):
    df = pd.read_csv(csv_path)
    df["dataset"] = df["dataset"].astype(str)
    df["model"] = df["model"].astype(str)
    df["rate"] = df["rate"].astype(int)
    qrels = Qrels.from_df(
        df=df,
        q_id_col="dataset",
        doc_id_col="model",
        score_col="rate",
    )

    return qrels


def load_runs(csv_path):
    df = pd.read_csv(csv_path)
    df["dataset"] = df["dataset"].astype(str)
    df["model"] = df["model"].astype(str)

    run = Run.from_df(
        df=df,
        q_id_col="dataset",
        doc_id_col="model",
        score_col="balanced_accuracy",
    )

    return run


def main():
    r_set = load_qrels('./Kaggle/Cold-Start original/Data/Ground_truth/groundtruth_0.5.csv')
    test = load_runs('./Kaggle/Experiment_part_2/result.csv')
    result = evaluate(r_set, test, metrics=["hit_rate@5", "hit_rate@10", "hit_rate@20",
                                   "precision@5", "precision@10", "precision@20",
                                   "recall@5", "recall@10", "recall@20",
                                   "ndcg@5", "ndcg@10", "ndcg@20",
                                   "mrr@5", "mrr@10", "mrr@20"])
    print(result)

if __name__ == '__main__':
    main()
