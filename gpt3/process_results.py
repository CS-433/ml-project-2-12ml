from sklearn import metrics
from run_gpt3 import prepare_copa, prepare_ecare, get_prompts_with_labels, parse_args
import pickle
import argparse


def get_gpt_predicted_label(gpt_prediction):
    # Get the predicted labels from GPT-3 generated text
    if "(a)" in gpt_prediction or "a)" in gpt_prediction or " a" == gpt_prediction[:2]:
        answer = 0
    elif "(b)" in gpt_prediction or "b)" in gpt_prediction or " b" == gpt_prediction[:2]:
        answer = 1
    else:
        raise ValueError("Text does not contain valid answer")
    return answer


def load_results(results_dir=".", k_shot=0, dataset="copa", explain=False):
    # lod pre-computed GPT-3 results
    results = {}
    for file_type in ["preds"]: # there could be many kinds of results, but only preds for now
        filename = f"{results_dir}/{'explain_' if explain else ''}gpt3_{dataset}_{file_type}_{k_shot}shot.bin"
        with open(filename, "rb") as f:
            results[file_type] = pickle.load(f)
    return results


def evaluate(gpt_preds, prompts_with_label):
    # the main evaluation loop
    y_true = []
    y_pred = []
    for i in range(len(gpt_preds)):
        try:
            pred = get_gpt_predicted_label(gpt_preds[i])
            y_pred.append(pred)
            y_true.append(prompts_with_label["label"][i])
        except ValueError:
            print(prompts_with_label[i]["prompt"])
            print("---")
            print(prompts_with_label[i])
            print(i)
            break
    classif_report = metrics.classification_report(y_true, y_pred)
    accuracy = metrics.accuracy_score(y_true, y_pred)
    return classif_report, accuracy


def evaluate_copa(results_dir, k_shot=0):
    train_set, dev_set = prepare_copa()
    prompts = get_prompts_with_labels(train_set, dev_set, k_shot, False)
    results = load_results(results_dir, k_shot=k_shot, dataset="copa", explain=False)
    classif_report, accuracy = evaluate(results["preds"], prompts)
    print(classif_report)
    print(f"Accuracy: {accuracy}")


def evaluate_ecare(results_dir, k_shot=0, explain=False):
    train_set, dev_set = prepare_ecare()
    prompts = get_prompts_with_labels(train_set, dev_set, k_shot, explain)
    results = load_results(results_dir, k_shot=k_shot, dataset="ecare", explain=explain)
    classif_report, accuracy = evaluate(results["preds"], prompts)
    print(classif_report)
    print(f"Accuracy: {accuracy}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k_shot", type=int, default=0)
    parser.add_argument("--explain", type=bool, default=False)
    parser.add_argument("--dataset", type=str, default="copa")
    parser.add_argument("--results_dir", type=str, default="results")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.dataset == "copa":
        evaluate_copa(args.results_dir, k_shot=args.k_shot)
    elif args.dataset == "ecare":
        evaluate_ecare(args.results_dir, k_shot=args.k_shot, explain=args.explain)
    else:
        raise NotImplementedError("Dataset not implemented")


if __name__ == "__main__":
    main()




