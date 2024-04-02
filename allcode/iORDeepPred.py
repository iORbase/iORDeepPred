import argparse
import all_script
from train_test import train_test

def main():
    parser = argparse.ArgumentParser(description='iORDeepPred')
    parser.add_argument("-d", "--data", action="store_true", help="Data preprocessing")
    parser.add_argument("-t", "--train", action="store_true", help="Training model")
    parser.add_argument("-p", "--predict", action="store_true", help="Predict the result")
    parser.add_argument("--csv_file", default="./csv_file/", help="Input data file path")
    parser.add_argument("--target_data", default="input_data", help="Model input file")
    parser.add_argument("--pretrain_flag", default=1, help="Use pretrained model or not")
    parser.add_argument("--pretrain_model", default="best_positive",help="Model file")
    parser.add_argument("--target_model", default="target",help="The name of the model saved after training")
    arg = parser.parse_args()

    if arg.data:
        print('start data preprocessing')
        protein_process = all_script.Protein_preprocessing(arg.csv_file)
        protein_process.main()
        smile_process = all_script.Smile_preprocessing(arg.csv_file)
        smile_process.main()
        all_script.Train_test_split(arg.target_data)
        print('model input file name ' + ': ' + arg.target_data)
        print('input data file path ' + ':' + arg.csv_file)
        print('finish data preprocessing')
    elif arg.train:
        print('start training model')
        train_test(arg.target_model, arg.pretrain_model, arg.train, arg.predict, arg.target_data, arg.pretrain_flag)
        print('finish training model')
    elif arg.predict:
        train_test(arg.target_model, arg.pretrain_model, arg.train, arg.predict, arg.target_data, arg.pretrain_flag)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
