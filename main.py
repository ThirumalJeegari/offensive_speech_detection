from data_preprocessing import preprocess_data
from model_training import train_logistic_regression, train_bert
from evaluation import evaluate_model

def main():
    X_train, X_test, y_train, y_test = preprocess_data('data/social_media_data.csv', 'label')
    
    print("Training Logistic Regression Model...")
    lr_model = train_logistic_regression(X_train, y_train)
    evaluate_model(lr_model, X_test, y_test, model_type='logistic_regression')
    
    print("Training BERT Model...")
    bert_model = train_bert(X_train, y_train)
    evaluate_model(bert_model, X_test, y_test, model_type='bert')

if __name__ == "__main__":
    main()
