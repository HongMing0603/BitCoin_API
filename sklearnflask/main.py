import sys
import os
import shutil
import time
import traceback

from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# inputs
training_data = './data/titanic.csv'
include = ['Age', 'Sex', 'Embarked', 'Survived']
# 依賴變量是我們的答案?Survived
dependent_variable = include[-1]

model_directory = 'model'
model_file_name = '%s/model.pkl' % model_directory
model_columns_file_name = '%s/model_columns.pkl' % model_directory

# These will be populated at training time
# 預測時填充
model_columns = None
clf = None

@app.route('/', methods=['GET'])
def hello_world():
    return "Hello world!!"


@app.route('/predict', methods=['POST'])
def predict():
    if clf:
        try:
            # 給他傳入json檔案(在調用網頁時)當作test_X
            json_ = request.json
            # get_dummies好像是會把文字類別的轉成OneHot形式
            # int ,float類型的維持原樣
            query = pd.get_dummies(pd.DataFrame(json_))
            # Age  Sex_female  Sex_male  Embarked_C  Embarked_S
        # 0   85           0         1           0           1
        # 1   24           1         0           1           0
        

            # https://github.com/amirziai/sklearnflask/issues/3
            # Thanks to @lorenzori
            query = query.reindex(columns=model_columns, fill_value=0)

            prediction = list(clf.predict(query))

            # Converting to int from int64
            # map(int,prediction) ->把prediction裡面的內容變成int類型?再放到list當中
            # 如果有小數點他就會無條件捨去變成整數
            return jsonify({"prediction": list(map(int, prediction))})
        # 有錯誤發生 傳回錯誤json格式
        except Exception as e:

            return jsonify({'error': str(e), 'trace': traceback.format_exc()})
    # 還沒有訓練模型
    else:
        print('train first')
        return 'no model here'


@app.route('/train', methods=['GET'])
def train():
    # using random forest as an example
    # can do the training separately and just update the pickles
    from sklearn.ensemble import RandomForestClassifier as rf
# 讀取titanic的csv
    df = pd.read_csv(training_data)
    # 挑選出你選取的欄位
    df_ = df[include]

    categoricals = []  # going to one-hot encode categorical variables
    # col是名稱
    # col_type是這個column的類型
    for col, col_type in df_.dtypes.items():
        
        if col_type == 'O':
            # 把這個名稱加入到categoricals裡邊
            # 最後結果是['Sex', 'Embarked']
            categoricals.append(col)
        else:
            df_[col].fillna(0, inplace=True)  # fill NA's with 0 for ints/floats, too generic
    # df 的col為:['Age', 'Sex', 'Embarked', 'Survived']
    
    # get_dummies effectively creates one-hot encoded variables
    # dummy_na ->連nan的部分都會顯示
    # 把我們剛剛紀錄的Categorical類別的column使用one hot 編碼進行轉換
    df_ohe = pd.get_dummies(df_, columns=categoricals, dummy_na=True)
    
# 把dependent_variable拿掉也就是Survived(是否存活 ->答案)
    x = df_ohe[df_ohe.columns.difference([dependent_variable])]
    y = df_ohe[dependent_variable]

    # capture a list of columns that will be used for prediction
    global model_columns
    model_columns = list(x.columns)
    # 把model 的column也就是x變成 pkl檔案放在資料夾中(model_columns_file_name)
    # pkl是常見的讀取模型檔案
    joblib.dump(model_columns, model_columns_file_name)

    global clf
    clf = rf()
    start = time.time()
    clf.fit(x, y)
    # 保存我們的模型
    joblib.dump(clf, model_file_name)

    message1 = 'Trained in %.5f seconds' % (time.time() - start)
    message2 = 'Model training score: %s' % clf.score(x, y)
    return_message = 'Success. \n{0}. \n{1}.'.format(message1, message2) 
    return return_message


@app.route('/wipe', methods=['GET'])
def wipe():
    try:
        shutil.rmtree('model')
        os.makedirs(model_directory)
        return 'Model wiped'

    except Exception as e:
        print(str(e))
        return 'Could not remove and recreate the model directory'


if __name__ == '__main__':
    try:
        # 第一個參數
        port = int(sys.argv[1])
    except Exception as e:
        port = 80


    try:
        clf = joblib.load(model_file_name)
        # 假如有model
        # ['Age' 'Embarked_C' 'Embarked_Q' 'Embarked_S' 'Embarked_nan' 'Sex_female'  'Sex_male' 'Sex_nan']
        print('model loaded')
        # 載入模型index
        # ['Age', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Embarked_nan', 'Sex_female', 'Sex_male', 'Sex_nan']
        # model_column.pkl
        model_columns = joblib.load(model_columns_file_name)
        print('model columns loaded')


# 找不到所以是第一次創建?
# 錯誤擷取 e是程式錯誤訊息
    except Exception as e:
        print('No model here')
        print('Train first')
        print(str(e))
        clf = None

    app.run(host='0.0.0.0', port=port, debug=True)
