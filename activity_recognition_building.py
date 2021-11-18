from header_imports import *


class activity_recognition_building(object):
    def __init__(self, model_type = "model1"):

        self.path = "Activity_Data/"
        self.training_data = pd.read_csv(self.path + "train.csv")
        self.test_data = pd.read_csv(self.path + "test.csv")
        self.Labels = None
        
        self.number_classes = 6
        self.model_summary = "model_summary/"
        self.save_path = "graph_charts/"

        self.model_type = model_type
        self.model = None
        
        self.data_dim = 562
        self.Labels = self.training_data['activity']
        self.Labels_keys = self.Labels.unique().tolist()
        self.Labels = np.array(self.Labels)


        self.X_train = 0
        self.X_test = 0
        self.Y_train = 0
        self.Y_test = 0
        
        self.label_data_normalize()  
        
        if self.model_type != "model5":
            self.input_shape = (self.X_train.shape[1], self.X_train.shape[2])
        
        if self.model_type == "model1":
            self.create_models_1()
        elif self.model_type == "model2":
            self.create_models_2()
        elif self.model_type == "model3":
            self.create_model_3()
        elif self.model_type == "model4":
            self.create_models_4()
        elif self.model_type == "model5":
            self.create_model_5()

        self.save_model_summary()


    def label_data_normalize(self):
        
        X_axis = self.training_data.drop("activity", axis=1)
        Y_axis = self.training_data["activity"].copy()
        Y_axis = pd.get_dummies(Y_axis)
        Y_axis = np.argmax(Y_axis.values, axis=1)

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X_axis.values, Y_axis, test_size = 0.10, random_state=42)

        Y_train_data = self.training_data.iloc[:,1]
        Y_training_data_count = np.array(Y_train_data.value_counts())
        activity = sorted(Y_train_data.unique())
        self.model_categories = activity
        
        self.Y_train_vec = tf.keras.utils.to_categorical(self.Y_train, self.number_classes)
        self.Y_test_vec = tf.keras.utils.to_categorical(self.Y_test, self.number_classes)

        plt.figure(figsize=(15,5))
        plt.pie(Y_training_data_count, labels = activity)
        plt.savefig(self.save_path + 'activity_percentage_data' +'.png', dpi =500)
        plt.clf()



    def create_models_1(self):

        self.model = Sequential()

        self.model.add(Conv2D(filters=32, kernel_size=3, activation="relu", input_shape = self.input_shape))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(self.number_classes, activation='softmax'))

        self.model.compile(loss = 'binary_crossentropy', optimizer ='adam', metrics= ['accuracy'])

        return self.model


    def create_models_2(self):

        self.model = Sequential()

        self.model.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu", input_dim = self.data_dim))
        self.model.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu"))

        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(rate=0.25))

        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))

        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(rate=0.25))
        self.model.add(Flatten())

        self.model.add(Dense(512, activation="relu"))
        self.model.add(Dropout(rate=0.5))
        self.model.add(Dense(units = self.number_classes, activation="softmax"))
        
        self.model.compile(loss = 'binary_crossentropy', optimizer ='adam', metrics= ['accuracy'])
	
        return self.model


    def create_model_3(self):

        self.model = Sequential()
        
        self.MyConv(first = True)
        self.MyConv()
        self.MyConv()
        self.MyConv()

        self.model.add(Flatten())
        self.model.add(Dense(units = self.number_classes, activation = "softmax", input_dim=2))

        self.model.compile(loss = "binary_crossentropy", optimizer ="adam", metrics= ["accuracy"])
        
        return self.model

        

    def MyConv(self, first = False):

        if first == False:
            self.model.add(Conv2D(filters=64, kernel_size=(4, 4),strides = (1,1), padding="same", input_dim = self.data_dim, activation="relu"))
        else:
            self.model.add(Conv2D(64,(4, 4),strides = (1,1), padding="same", input_dim = self.data_dim, activation="relu"))
    
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.5))

        self.model.add(Conv2D(filters=32, kernel_size = (4, 4),strides = (1,1),padding="same"))
        self.model.add(Activation("relu"))
        self.model.add(Dropout(0.25))

        

    def create_models_4(self):
        
        self.model = Sequential()

        self.model.add(LSTM(16,  input_dim = self.data_dim, activation="relu"))
        self.model.add(Dense(1, activation='relu'))
        self.model.add(Dense(self.number_classes, activation='softmax'))

        return self.model



    def create_model_5(self):
        
        self.model = Sequential()
        
        self.model.add(Dense(32,activation='sigmoid',input_dim=self.data_dim))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(6,activation='softmax'))
        self.model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    # Save the model summery as a txt file
    def save_model_summary(self):
        with open(self.model_summary + self.model_type +"_summary_architecture_" + ".txt", "w+") as model:
            with redirect_stdout(model):
                self.model.summary()






