from flask import Flask, render_template, request
import pandas as pd
import joblib as jb
import numpy as np

app = Flask(__name__)

# def calculate_rmse(predicted_ratings, actual_ratings):
#     diff = predicted_ratings - actual_ratings
#     rmse = np.sqrt(np.mean(diff**2))
#     return rmse

def calculate_rmse(predicted_ratings, actual_ratings):
    diff = predicted_ratings - actual_ratings
    rmse = np.sqrt(np.mean(diff**2))
    return rmse

def grupRating(metode, user_id):
    rating_cols = ["user_id", "movie_id", "rating", "unix_timestamp"]
    test_data = pd.read_csv(
    "https://raw.githubusercontent.com/ThoriqFathu/SRP/master/ml-100k/u"
    + str(2)
    + ".test",
    sep="\t",
    names=rating_cols,
    )

    test_data.drop("unix_timestamp", inplace=True, axis=1)
    # rating_cols = ['user_id', 'movie_id', 'rating']
    # test_data = pd.read_csv('https://raw.githubusercontent.com/ThoriqFathu/SRP/master/Cold-Start/tes-fold'+str(2)+'-rating10.txt', sep='::', names=rating_cols)
    data_ = test_data.values
    test_data = np.array(test_data)
    # st.write(test_data[0][0] == 1)
    model = jb.load(f"models/{metode}.joblib")
    rat_user = []
    item_test = []
    for i in test_data:
        if i[0] == user_id:
            rat_user.append(i)
            item_test.append(i[1])
    # a, b, c = rat_user[0]
    # st.write(a)
    # st.write(rat_user[0])
    # st.write(user_id)
    # model = jb.load("modelitr2.joblib")
    matClus, anggotaClus, weights = model
    for ind, g in enumerate(anggotaClus):
        if (user_id - 1) in g:
            nearsGrup = ind
    userIndex = anggotaClus[nearsGrup].index(user_id - 1)
    vek_rating = matClus[nearsGrup][userIndex]
    list_pred = []
    pred_ratings = []
    item_train = []
    for k, data in enumerate(vek_rating):
        # st.write(data)
        if data == 0:
            item_id = k + 1
            mat = np.array((matClus)[nearsGrup])[:, item_id - 1]
            # # print(np.array(mat))
            # index_u = np.where(mat!=0)[0]
            # -------------------

            # print((index_u))
            # print(userIndex)
            # print(nearsGrup)
            # print(len(anggotaClus[0]))
            # print(weights[nearsGrup][8][userIndex])
            # print(np.array(index_u))

            # ----------------------
            influence = np.zeros(len(mat))
            for uID in range(len(mat)):
                index_u = np.where(mat != 0)[0]

                # print(uID, userIndex)
                if len(index_u) == 0:
                    pred = 1
                else:
                    index_u = index_u[index_u != uID]
                    uweight = []
                    for index in index_u:
                        if np.isnan(weights[nearsGrup][index][uID]):
                            uweight.append(0)
                        else:
                            uweight.append(
                                np.nan_to_num(
                                    weights[nearsGrup][index][uID]
                                )
                            )
                        # print(uweight)
                        # uweight = np.nan_to_num(uweight)
                    if mat[uID] == 0:
                        r = 1
                    else:
                        r = mat[uID]
                    if len(uweight) != 0:
                        pred = sum((mat[index_u] - r) * np.array(uweight))
                        pred += r
                    else:
                        pred = r
                    if pred > 5:
                        pred = 5
                    if pred < 1:
                        pred = 1
                influence[uID] = pred
            # print('item - ',item_id-1)
            # print(influence)
            pred = np.mean(influence)
            if int(item_id) in item_test:
                pred_ratings.append(pred)
            list_pred.append([int(item_id), pred])
            # st.write(item_id)
        else:
            item_id = k + 1
            item_train.append([int(item_id), int(data)])

    # list_pred =  dict(sorted(list_pred.items(), key=lambda item: item[1], reverse=True))
    list_pred = sorted(list_pred, key=lambda x: x[1], reverse=True)
    # st.write(list_pred)
    return list_pred, item_train, rat_user, pred_ratings

def influence(metode, user_id):
    rating_cols = ["user_id", "movie_id", "rating", "unix_timestamp"]
    test_data = pd.read_csv(
    "https://raw.githubusercontent.com/ThoriqFathu/SRP/master/ml-100k/u"
    + str(2)
    + ".test",
    sep="\t",
    names=rating_cols,
    )

    test_data.drop("unix_timestamp", inplace=True, axis=1)
    data_ = test_data.values
    test_data = np.array(test_data)

    model = jb.load(f"models/{metode}.joblib")
    rat_user = []
    item_test = []
    for i in test_data:
        if i[0] == user_id:
            rat_user.append(i)
            item_test.append(i[1])
    # a, b, c = rat_user[0]
    # st.write(a)
    # st.write(rat_user[1])
    # st.write(user_id)
    # model = jb.load("modelitr2.joblib")
    matClus, anggotaClus, weights = model
    for ind, g in enumerate(anggotaClus):
        if (user_id - 1) in g:
            nearsGrup = ind
    userIndex = anggotaClus[nearsGrup].index(user_id - 1)
    vek_rating = matClus[nearsGrup][userIndex]
    list_pred = []
    pred_ratings = []
    item_train = []
    for k, data in enumerate(vek_rating):
        # st.write(data)
        if data == 0:
            item_id = k + 1
            # st.write(nearsGrup)
            # st.write(userIndex)
            mat = np.array((matClus)[nearsGrup])[:, item_id - 1]
            # print(np.array(mat))
            index_u = np.where(mat != 0)[0]
            if len(index_u) == 0:
                pred = 1.0
            else:
                uweight = []
                for index in index_u:
                    uweight.append(weights[nearsGrup][index][userIndex])
                # print(uweight)
                # uweight = np.nan_to_num(uweight)
                pred = sum((mat[index_u] - 1) * np.array(uweight))
                pred += 1
                if pred > 5:
                    pred = 5.0
                if pred < 1:
                    pred = 1.0
            # list_pred[int(item_id)] = pred
            if int(item_id) in item_test:
                pred_ratings.append(pred)
            list_pred.append([int(item_id), pred])
            # st.write(item_id)
        else:
            item_id = k + 1
            item_train.append([int(item_id), int(data)])

    # st.write(userIndex,nearsGrup)
    # st.write(anggotaClus[1][0])
    # list_pred =  dict(sorted(list_pred.items(), key=lambda item: item[1], reverse=True))
    list_pred = sorted(list_pred, key=lambda x: x[1], reverse=True)
    return list_pred, item_train, rat_user, pred_ratings

@app.route('/rekomendasifilm/', methods=['GET', 'POST'])
def rekomendasi_film():
    if request.method == 'POST':
        step = request.form['step']
        if step == "pilih":
            metode = request.form['metode']
            metode_full = request.form['metode_full']

            rating_cols = ["user_id", "movie_id", "rating", "unix_timestamp"]
            test_data = pd.read_csv(
            "https://raw.githubusercontent.com/ThoriqFathu/SRP/master/ml-100k/u"
            + str(2)
            + ".test",
            sep="\t",
            names=rating_cols,
            )

            test_data.drop("unix_timestamp", inplace=True, axis=1)
            # rating_cols = ['user_id', 'movie_id', 'rating']
            # test_data = pd.read_csv('https://raw.githubusercontent.com/ThoriqFathu/SRP/master/Cold-Start/tes-fold'+str(2)+'-rating10.txt', sep='::', names=rating_cols)
            data_ = test_data.values
            test_data = np.array(test_data)

            users_id = [data[0] for data in test_data]
            target = list(set(users_id))

            top_n = 100
            listtop = [x + 1 for x in range(top_n)]

            return render_template('rekomendasi.html', target=target, jumlah_n=listtop, metode=metode, metode_full=metode_full )
        
        elif step == "rekom":
            metode = request.form['metode']
            metode_full = request.form['metode_full']
            top_n = request.form['top_n']
            top_n = int(top_n)
            loop = range(1, top_n + 1)
            user_target = request.form['user_target']
            based = request.form['based']
            user_target = int(user_target)

            if based == 'influence-based':
                list_pred, item_train, rat_user, pred_ratings = influence(metode, user_target)
            else:
                list_pred, item_train, rat_user, pred_ratings = grupRating(metode, user_target)
            
            columns = [
                "ID",
                "Title",
                "Release Date",
                "URL",
                "Col1",
                "Col2",
                "Col3",
                "Col4",
                "Col5",
                "Col6",
                "Col7",
                "Col8",
                "Col9",
                "Col10",
                "Col11",
                "Col12",
                "Col13",
                "Col14",
                "Col15",
                "Col16",
                "Col17",
                "Col18",
                "Col19",
                "Col20",
            ]
            df_item = pd.read_csv(
                "https://raw.githubusercontent.com/ThoriqFathu/SRP/master/ml-100k/u.item",
                sep="|",
                header=None,
                names=columns,
                encoding="latin-1",
            )

            list_recom = []
            id_recom = []
            for i in range(top_n):
                if i < len(list_pred):
                    row_item = df_item.loc[df_item["ID"] == list_pred[i][0]]
                    judul = row_item["Title"].values
                    list_recom.append([list_pred[i][0], judul[0], list_pred[i][1]])
                    id_recom.append(int(list_pred[i][0]))

            list_train = []
            for i in range(len(item_train)):
                if i < len(list_pred):
                    row_item = df_item.loc[df_item["ID"] == item_train[i][0]]
                    judul = row_item["Title"].values
                    list_train.append([judul[0], item_train[i][0], item_train[i][1]])

            list_ground = []
            id_ground = []
            for i in range(len(rat_user)):
                if i < len(list_pred):
                    row_item = df_item.loc[df_item["ID"] == rat_user[i][1]]
                    judul = row_item["Title"].values
                    list_ground.append([rat_user[i][1], judul[0], rat_user[i][2]])
                    id_ground.append(int(rat_user[i][1]))

            list_irisan = []
            irisan = list(set(id_recom) & set(id_ground))
            for i in range(top_n):
                if i < len(irisan):
                    row_item = df_item.loc[df_item["ID"] == irisan[i]]
                    judul = row_item["Title"].values
                    for k in range(len(list_ground)):
                        acr = int(list_ground[k][0])
                        if acr == irisan[i]:
                            list_irisan.append([irisan[i], judul[0], list_ground[k][2], list_pred[i][1]])
            
            presisi = (len(irisan)) / (top_n)
            # actual_ratings = np.array([data[2] for data in rat_user])
            # pred_ratings = np.array(pred_ratings)

            actual_ratings = np.array([data[2] for data in rat_user if int(data[1]) in irisan])
            pred_ratings = np.array([data[1] for data in list_pred if int(data[0]) in irisan])
            
            rmse = calculate_rmse(pred_ratings, actual_ratings)

            return render_template('hasil_rekomendasi.html', user_target=user_target, top_n=top_n, metode=metode, metode_full=metode_full, list_recom=list_recom, list_train=list_train, list_ground=list_ground, list_irisan=list_irisan, presisi=presisi, rmse=rmse)
            # return rmse

        elif step == "detail":
            metode_full = request.form['metode_full']
            top_n = request.form['top_n']
            user_target = request.form['user_target']
            list_recom_str = request.form.get('list_recom')
            list_recom = eval(list_recom_str)
            list_train_str = request.form.get('list_train')
            list_train = eval(list_train_str)
            list_ground_str = request.form.get('list_ground')
            list_ground = eval(list_ground_str)
            list_irisan_str = request.form.get('list_irisan')
            list_irisan = eval(list_irisan_str)
            presisi = request.form['presisi']
            presisi = float(presisi)
            rmse = request.form['rmse']
            rmse = float(rmse)


            return render_template('detail.html', user_target=user_target, top_n=top_n, metode_full=metode_full, list_recom=list_recom,  list_train=list_train, list_ground=list_ground, list_irisan=list_irisan, presisi=presisi, rmse=rmse)
            

        

    return render_template('index.html')

@app.route('/rekomendasifilm/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)