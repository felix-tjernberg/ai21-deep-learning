from sklearn.model_selection import train_test_split


def create_train_test_eval_split(X, y, test_size=0.3, eval_size=0.3):
    "Returns tuple with full_split and eval_split dictionaries"
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    X_train_eval, X_test_eval, y_train_eval, y_test_eval = train_test_split(
        X_train_full, y_train_full, test_size=eval_size, random_state=42
    )
    return (
        {
            "X_train": X_train_full,
            "X_test": X_test_full,
            "y_train": y_train_full,
            "y_test": y_test_full,
        },
        {
            "X_train": X_train_eval,
            "X_test": X_test_eval,
            "y_train": y_train_eval,
            "y_test": y_test_eval,
        },
    )
