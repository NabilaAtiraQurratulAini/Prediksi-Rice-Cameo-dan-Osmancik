import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
from sklearn.svm import SVC

st.title("""PREDIKSI PADI CAMEO AND OSMANCIK""")
st.write("Nabila Atira Qurratul Aini")

tabs = st.tabs(["Business Understanding", "Data Understanding", "Data Preprocessing", "Modeling", "Deployment", "Informasi"])
business_understanding, data_understanding, data_preprocessing, modeling, deployment, informasi = tabs

with business_understanding:
    st.write("# BUSINESS UNDERSTANDING")
    
    st.write("### Latar Belakang")
    st.write("Industri pertanian memiliki peranan penting dalam perekonomian Turki, khususnya dalam produksi padi bersertifikat. Dua varietas padi yang signifikan di Turki adalah Osmancik dan Cammeo, yang masing-masing memiliki karakteristik unik. Identifikasi yang tepat dan pemahaman yang mendalam tentang kedua varietas ini sangat penting untuk meningkatkan efisiensi pertanian dan kualitas hasil panen.")
    
    st.write("### Tujuan")
    st.write("Tujuan dari penelitian ini adalah untuk meningkatkan pemahaman dan klasifikasi terhadap dua spesies padi bersertifikat di Turki, yaitu Osmancik dan Cammeo. Dengan memiliki pemahaman yang lebih baik tentang karakteristik dan perbedaan antara kedua spesies ini, kita dapat mendukung pengembangan kebijakan pertanian yang lebih efektif dan meningkatkan produktivitas tanaman padi bersertifikat di Turki.")
    
    st.write("### Manfaat")
    st.write("Penelitian ini diharapkan dapat meningkatkan akurasi identifikasi varietas padi Osmancik dan Cammeo, yang akan mengurangi kesalahan dan meningkatkan efisiensi operasional. Selain itu, data yang dihasilkan dapat membantu menyusun kebijakan pertanian yang lebih efektif, mendukung pengelolaan lahan dan penanaman yang lebih baik. Pemahaman yang lebih baik tentang karakteristik masing-masing varietas juga memungkinkan optimalisasi praktik pertanian, meningkatkan produktivitas dan kualitas hasil panen, serta kepuasan pelanggan dan reputasi produk padi Turki di pasar.")
    
with data_understanding:
    st.write("# DATA UNDERSTANDING")

    st.write("### Pemahaman Data")
    st.write("Dataset terdiri dari 3810 gambar butiran beras untuk spesies Osmancik dan Cammeo. Karakteristik umum spesies Osmancik mencakup penampilan yang lebar, panjang, mengkilap, dan kusam, sementara spesies Cammeo memiliki ciri serupa dengan penampilan yang lebar dan panjang, serta kecenderungan mengkilap dan kusam. Terdapat 7 fitur morfologis untuk setiap butir beras, yaitu luas, keliling, panjang sumbu utama, panjang sumbu minor, eksentrisitas, luas cembung, dan konveksitas. Data ini dikategorikan menjadi satu fitur kelas yang menyimpan informasi tentang spesies butir beras. Fitur-fitur tersebut diekstraksi dari gambar yang diperoleh melalui serangkaian langkah pengolahan gambar.")
    st.write("- Area atau Daerah : Fitur ini mengukur luas dari objek. Luas dapat dihitung dalam unit piksel atau unit luas lainnya, tergantung pada resolusi data. Luas memberikan informasi tentang ukuran relatif objek. Data yang terkait dengan fitur ini memiliki tipe data numerik.")
    st.write("- Perimeter : Perimeter mengukur panjang garis batas objek. Ini diukur sebagai jumlah panjang semua tepi objek. Perimeter bisa memberikan indikasi seberapa kompleks bentuk objek tersebut. Data yang terkait dengan fitur ini memiliki tipe data numerik.")
    st.write("- Major Axis Length atau Panjang Sumbu Utama : Sumbu utama adalah sumbu terpanjang dalam elips yang mengelilingi objek. Panjang sumbu ini memberikan gambaran tentang dimensi utama objek dan arah orientasi elips. Data yang terkait dengan fitur ini memiliki tipe data numerik.")
    st.write("- Minor Axis Length atau Panjang Sumbu Kecil : Sumbu kecil adalah sumbu terpendek dalam elips yang mengelilingi objek. Ini memberikan informasi tentang dimensi kedua objek dan dapat membantu menggambarkan bentuk elips. Data yang terkait dengan fitur ini memiliki tipe data numerik.")
    st.write("- Eccentricity atau Eksentrisitas : Eksentrisitas mengukur sejauh mana elips yang mengelilingi objek mendekati bentuk lingkaran. Nilai eksentrisitas 0 menunjukkan objek yang bentuknya mendekati lingkaran sempurna, sementara nilai mendekati 1 menunjukkan elips yang sangat panjang. Data yang terkait dengan fitur ini memiliki tipe data numerik.")
    st.write("- Convex Area atau Daerah Cembung : Luas cembung mengukur luas daerah yang diukur dari cembung (convex hull) objek. Cembung adalah poligon terkecil yang dapat mencakup seluruh objek. Luas ini memberikan informasi tentang sejauh mana objek dapat dianggap 'cembung'. Data yang terkait dengan fitur ini memiliki tipe data numerik.")
    st.write("- Extent atau Luas : Extent adalah rasio antara luas objek dan luas kotak terkecil yang dapat mengelilingi objek. Nilai 1 menunjukkan objek yang mengisi kotak dengan sempurna, sementara nilai yang lebih rendah menunjukkan objek yang mungkin memiliki bentuk yang lebih tidak teratur. Data yang terkait dengan fitur ini memiliki tipe data numerik.")
    st.write("- Class atau Kelas : Kelas adalah label kategori atau jenis keanggotaan dari objek. Ini adalah informasi klasifikasi yang menunjukkan keberadaan objek dalam kategori tertentu, seperti 0 : kelas 'Rice Cammeo' atau 1 : kelas 'Rice Osmancik'. Jumlah data untuk Cammeo adalah 1630, dan jumlah data untuk kelas Osmancik adalah 2180.")
    
    df = pd.read_csv("https://raw.githubusercontent.com/NabilaAtiraQurratulAini/PsdA/main/Rice_Osmancik_Cammeo_Dataset.csv")
    st.write("### Dataset Rice Cameo and Osmancik")
    df

    st.write("### Informasi Dataset")
    informasi_df = pd.DataFrame({'Column': df.columns, 'Non-Null Count': [df[col].notnull().sum() for col in df.columns], 'Dtype': [df[col].dtype for col in df.columns]})
    st.dataframe(informasi_df)

with data_preprocessing:
    st.write("# DATA PREPROCESSING")

    st.write("### Data Missing Value")
    null_counts = df.isnull().sum().reset_index()
    null_counts.columns = ['Column', 'Null Count']
    st.dataframe(null_counts)

    st.write("### Data Duplikat")
    num_duplicates = df.duplicated().sum()
    st.write('Jumlah baris duplikat :', num_duplicates)

    st.write("### Blox Plot")
    fig, axs = plt.subplots(1, 7, figsize=(20, 6))
    features = ['AREA', 'PERIMETER', 'MAJORAXIS', 'MINORAXIS', 'ECCENTRICITY', 'CONVEX_AREA', 'EXTENT']
    for i, feature in enumerate(features):
        axs[i].boxplot(df[feature])
        axs[i].set_title(feature)
        axs[i].set_ylabel('Value')
        axs[i].set_xticklabels([feature], rotation=45)
    fig.suptitle('Boxplots of Features')
    st.pyplot(fig)

    st.write("### Data Outlier")
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    z_scores = np.abs(zscore(df[numeric_columns]))
    threshold = 3
    outliers = (z_scores > threshold).any(axis=1)
    jumlah_outlier = outliers.sum()
    jumlah_tanpa_outlier = len(df) - jumlah_outlier
    st.write("Jumlah outlier :", jumlah_outlier)
    st.write("Jumlah data tanpa outlier :", jumlah_tanpa_outlier)

    st.write("### Data Bersih")
    data = df[~outliers]
    data

    fig, axs = plt.subplots(1, 7, figsize=(20, 6))
    features = ['AREA', 'PERIMETER', 'MAJORAXIS', 'MINORAXIS', 'ECCENTRICITY', 'CONVEX_AREA', 'EXTENT']
    for i, feature in enumerate(features):
        axs[i].boxplot(data[feature])
        axs[i].set_title(feature)
        axs[i].set_ylabel('Value')
        axs[i].set_xticklabels([feature], rotation=45)
    fig.suptitle('Boxplots of Features')
    st.pyplot(fig)

    st.write("### Statistik Deskriptif")
    deskripsi_df = df.describe()
    st.dataframe(deskripsi_df)

    st.write("### Diagram Batang")
    class_counts = df['CLASS'].value_counts()
    plt.figure(figsize=(8, 6))
    class_counts.plot(kind='bar', color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.title('Frequency of Class Variable')
    st.pyplot(plt)

    st.write("### Encoding")
    data['CLASS'].replace('Cammeo', 0,inplace=True)
    data['CLASS'].replace('Osmancik', 1,inplace=True)
    st.write("DataFrame setelah binary encoding :")
    st.dataframe(data)

    st.write("### Korelasi Matriks")
    kolerasi = data[['AREA', 'PERIMETER', 'MAJORAXIS', 'MINORAXIS', 'ECCENTRICITY', 'CONVEX_AREA', 'EXTENT']]
    correlation_matrix = kolerasi.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix')
    st.pyplot(plt)

    st.write("### Normalisasi Data")
    st.write("Normalisasi data adalah proses untuk mengubah rentang data sehingga memiliki skala yang konsisten, yang penting untuk banyak algoritma machine learning. Salah satu metode normalisasi yang umum digunakan adalah Standard Scaler.")
    st.markdown("""
    Standard Scaler, juga dikenal sebagai Z-score normalization atau standar deviasi normalisasi, bekerja dengan cara berikut :
    - Mean-Centering (Pengaturan Rata-Rata ke Nol) : Data diubah sehingga rata-ratanya (mean) menjadi nol. Ini dilakukan dengan mengurangkan rata-rata dari setiap nilai data.
    - Scaling (Pengaturan Variansi) : Setelah mean diatur ke nol, data juga diubah sehingga memiliki deviasi standar (standard deviation) sebesar satu. Ini dicapai dengan membagi deviasi setiap nilai dari rata-rata dengan deviasi standar data.
    """)
    
    st.write("### Rumus Z-Score")
    st.write("Rumus untuk normalisasi menggunakan Standard Scaler adalah : ")
    st.latex(r"""
    z = \frac{\sigma}{x - \mu}
    """)

    st.markdown(r"""
    Keterangan :
    - 𝑧 adalah Z-score dari nilai individual. Z-score menunjukkan seberapa jauh dan dalam arah apa (positif atau negatif) sebuah nilai berada dari rata-rata dataset dalam satuan deviasi standar. Z-score adalah hasil akhir dari normalisasi.
    - 𝜎 adalah Deviasi standar (standard deviation) dari dataset. Deviasi standar mengukur sebaran data di sekitar rata-rata. Semakin besar deviasi standar, semakin besar penyebaran data.
    - 𝑥 adalah Nilai individual dari data yang ingin dinormalisasi. Ini adalah nilai asli yang akan dikonversi menjadi Z-score.
    - 𝜇 adalah Rata-rata (mean) dari dataset. Rata-rata adalah jumlah semua nilai data dibagi dengan jumlah total data. Ini adalah pusat distribusi data.
    """)

    st.write("### Dataset Hasil Normalisasi")
    scaler = StandardScaler()
    fitur = ["AREA", "PERIMETER", "MAJORAXIS", "MINORAXIS", "ECCENTRICITY", "CONVEX_AREA", "EXTENT"]
    data.loc[:, fitur] = scaler.fit_transform(data[fitur])
    data

    st.write("### Label Data CLASS")
    class_counts = df['CLASS'].value_counts().reset_index()
    class_counts.columns = ['Class', 'Frequency']
    st.dataframe(class_counts)

    st.write("### Split Data")
    X = data.drop(columns=['CLASS'])
    y = data['CLASS']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_size = X_train.shape[0]
    st.write('Jumlah data train :', train_size)
    test_size = X_test.shape[0]
    st.write('Jumlah data test :', test_size)


with modeling:
    st.write("# MODELING")
    
    st.write("### Metode Support Vector Machine")
    st.write("Support Vector Machine (SVM) adalah algoritma pembelajaran mesin yang digunakan untuk klasifikasi dan regresi. SVM bekerja dengan cara mencari garis atau hyperplane yang memisahkan data ke dalam kelas-kelas berbeda dengan margin maksimum. Konsep utamanya adalah membuat keputusan yang optimal dengan memaksimalkan margin pemisahan antara kelas-kelas data.")
    
    st.write("### Rumus Support Vector Machine")
    st.write("Dalam Support Vector Machine (SVM), fungsi keputusan untuk klasifikasi dinyatakan dengan :")
    st.latex(r"""
    f(x) = w \cdot x + b
    """)
    
    st.markdown(r"""
    Keterangan :
    - 𝑓(𝑥) adalah nilai prediksi untuk data input 𝑥.
    - 𝑤 adalah vektor bobot yang telah ditentukan selama pelatihan SVM. Vektor ini menentukan orientasi hyperplane pemisah.
    - 𝑥 adalah vektor fitur dari data yang ingin diklasifikasikan.
    - 𝑏 adalah bias yang juga diperoleh selama pelatihan SVM. Bias ini menentukan jarak hyperplane dari asal koordinat.
    """)

    st.write("Klasifikasi Data :")
    st.markdown(r"""
    Hasil prediksi 𝑓(𝑥) digunakan untuk menetukan kelas data sebagai berikut :
    - Jika 𝑓(𝑥) > 0, maka data akan diklasifikasikan ke dalam kelas positif (misalnya,kelas 1).
    - Jika 𝑓(𝑥) < 0, maka data akan diklasifikasikan ke dalam kelas negatif (misalnya, kelas -1 atau 0, tergantung pada penetapan kelas).
    """)

    st.markdown(r"""
    Dalam implementasi praktis, biasanya digunakan nilai ambang (threshold) 0, artinya :
    - Jika 𝑓(𝑥) > 0, data diklasifikasikan ke dalam kelas positif (kelas 1).
    - Jika 𝑓(𝑥) < 0, data diklasifikasikan ke dalam kelas negatif (kelas 0).
    """)
    
    svm_model = SVC(C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=42)
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)

    st.write("### Matriks Evaluasi")
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    precision_svm = precision_score(y_test, y_pred_svm)
    recall_svm = recall_score(y_test, y_pred_svm)
    f1_svm = f1_score(y_test, y_pred_svm)
    metrics_df = pd.DataFrame({'Akurasi': [accuracy_svm], 'Precision': [precision_svm], 'Recall': [recall_svm], 'F1-Score': [f1_svm]})
    metrics_df

    st.write("### Tabel Prediksi")
    svm_results_data = pd.DataFrame({'Actual Label': y_test, 'Prediksi SVM': y_pred_svm})
    svm_results_data

    st.write("### Confusion Matriks")
    confusion_matrix_svm = confusion_matrix(y_test, y_pred_svm)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(confusion_matrix_svm, annot=True, fmt='d', cmap='Reds', xticklabels=svm_model.classes_, yticklabels=svm_model.classes_)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('Actual Label')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)


with deployment:
    st.write("### APLIKASI PREDIKSI JENIS PADI")

    AREA = st.number_input("Masukkan Nilai Area : ")
    PERIMETER = st.number_input("Masukkan Nilai Perimeter : ")
    MAJORAXIS = st.number_input("Masukkan Nilai Majoraxis : ")
    MINORAXIS = st.number_input("Masukkan Nilai Minoraxis : ")
    ECCENTRICITY = st.number_input("Masukkan Nilai Eccentricity : ")
    CONVEX_AREA = st.number_input("Masukkan Nilai Convex Area : ")
    EXTENT = st.number_input("Masukkan Nilai Extent : ")

    if st.button("Prediksi"):
        new_data = np.array([[AREA, PERIMETER, MAJORAXIS, MINORAXIS, ECCENTRICITY, CONVEX_AREA, EXTENT]])
        new_data = scaler.transform(new_data)
        prediction = svm_model.predict(new_data)
        if prediction[0] == 1:
            st.write("Hasil Prediksi : Beras Osmancik")
        else:
            st.write("Hasil Prediksi : Beras Cammeo")

with informasi:
    st.write("# INFORMASI DATASET")
    
    st.write("### Sumber Dataset di UCI")
    st.write("https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik")

    st.write("### Source Code di Github")
    st.write("https://github.com/NabilaAtiraQurratulAini/Prediksi-Rice-Cameo-dan-Osmancik.git")
