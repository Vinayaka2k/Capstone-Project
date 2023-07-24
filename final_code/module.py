import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense,LeakyReLU
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.layers import LayerNormalization
from keras.layers.embeddings import Embedding
from keras.optimizers import *
from keras.layers import Input
from keras.models import Model
from keras.models import model_from_json

##################################################################
paddingChar = 0.88
maxLength = 3
encodingDictonary = dict()
decodingDictonary = dict()
decodingDictonary1 = dict()
listOfEncodedValues = []

optimizerd = Adam(lr=0.001)
optimizer1 = SGD()
optimizergan = Adam(lr=0.0001)
epochs = 20
noise_dim = 2
batch_size = 1
    
####################################################################

def init():
    createDictonary("encoding_dict.txt")
    createDictonaryDecode("decoding_dict.txt")

#####################################################################

def createDictonary(fileUrl):
    file = open(fileUrl,'r')
    reader = file.read()
    reader = reader.split("\n")
    for r in(reader):
        row = r.split(" ")
        newKey = row[0]
        newValue = float(row[1])
        encodingDictonary[newKey] = newValue
        decodingDictonary1[newValue] = newKey
        listOfEncodedValues.append(newValue)
    file.close()


def createDictonaryDecode(fileUrl):
    file = open(fileUrl,'r')
    reader = file.read()
    reader = reader.split("\n")
    for r in(reader):
        row = r.split(" ")
        newKey = row[0]
        newValue = int(row[1])        
        decodingDictonary[newValue] = newKey
        
        #listOfEncodedValues.append(newValue)
    file.close()


def encodeString(inputString):
    outputList = []
    lengthOfString = len(inputString)
    for c in inputString:
        outputList.append(encodingDictonary[c])
    for i in range(lengthOfString,maxLength+1):
        outputList.append(paddingChar)
    return outputList

def decodeList(inputList):
    decodedString = ""
    for i in inputList:
        if(i == 0):
            break
        decodedString = decodedString + str(decodingDictonary[get_int_encoded(i)])
    return decodedString


def get_int_encoded(number):
  number *= 1000
  number = int(number)
  #print("n:",number)
  temp = number
  if(number%20 != 0):
    base_number = number//20
    base_number*=20
    if((number-base_number) <= 10 ):
			#return round(number/1000,2)
      temp = base_number
    else:
			#return round((number+20)/1000,2)
      temp = base_number+20
  temp = temp//10
  if temp>88:
    return 88
  if temp<-98:
    return -98
  return temp

####################################################################3
def get_discriminator_model(model_name):
    json_file = open('./trained_d/'+model_name+".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    discriminator = model_from_json(loaded_model_json)
    discriminator.load_weights("./trained_d/"+model_name+".h5")
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizerd)
    return discriminator

def generate_passwords(generator, size):    
    res=[]
    noise = np.random.uniform(-0.98, 0.88, size=(int(size), noise_dim))
    fake_x = generator.predict(noise)
    for i in fake_x:
        res.append(decodeList(i))
    return res

def get_generator_model(model):
    res = dict()
    model_list = os.listdir("./trained_g")
    for file in model_list:
        if model in file:
            if file.split(".")[1] == "json":
                res["model"] = file
            else:
                res["weights"] = file
    return res

def generate_new_passwords(model_name,size):
    dict = get_generator_model(model_name)
    model = dict["model"]
    weights = dict["weights"]

    json_file = open('./trained_g/'+model, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    generator = model_from_json(loaded_model_json)
    generator.load_weights("./trained_g/"+weights)
    generator.compile(loss='binary_crossentropy', optimizer=optimizerd)
    return generate_passwords(generator, size)

#################################################################################

def check_strength_password(model_name, password):
    discriminator = get_discriminator_model(model_name)
    temp_list = []
    for i in password:
        temp_list.append(encodingDictonary[i])
    for i in range(maxLength - len(password) + 1):
        temp_list.append(paddingChar)
    temp_list = np.array(temp_list)
    temp_list = temp_list.reshape(1,maxLength+1)
    password_strength = discriminator.predict_on_batch(temp_list)
    password_strength = 1-password_strength
    
    if password_strength < 0.65:
        strength = "Weak"
    elif password_strength >= 0.65 and password_strength < 0.8:
        strength = "Average"
    else:
        strength = "Strong"
    return strength+" "+str(password_strength)

##########################################################################

def save_model(model_name, generator, discriminator, gan):
    generator_json = generator.to_json()
    with open("./trained_g/"+model_name+".json", "w") as json_file:
        json_file.write(generator_json)
    generator.save_weights("./trained_g/"+model_name+".h5")
    
    discriminator_json = discriminator.to_json()
    with open("./trained_d/"+model_name+".json", "w") as json_file:
        json_file.write(discriminator_json)
    discriminator.save_weights("./trained_d/"+model_name+".h5")
    
    gan_json = gan.to_json()
    with open("./trained_gan/"+model_name+".json", "w") as json_file:
        json_file.write(gan_json)
    gan.save_weights("./trained_gan/"+model_name+".h5")

def round_off(ele):
  return round(ele)

func = np.vectorize(round_off)

def create_generator(input_size):
    generator = Sequential()   
    generator.add(Dense(4, input_dim=noise_dim))
    #generator.add(())
    #generator.add(Dense(9))
    #generator.add(BatchNormalization())
    generator.add(Dense(input_size, activation='tanh'))
    #generator.compile(loss='mse', optimizer=optimizer1)
    generator.compile()
    return generator

def create_discriminator(input_size):
    discriminator = Sequential()
    discriminator.add(Dense(4, input_dim=input_size))
    #discriminator.add(Dense(5, input_dim=input_size))
    discriminator.add(BatchNormalization())
    #discriminator.add(Dense(1, activation='sigmoid'))
    discriminator.add(Dense(1, activation='sigmoid'))
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizerd)
    return discriminator

def train(input_file, model_name):
    file = open('./data/'+input_file,"r")
    valid=file.read()
    labels = []
    no_valid_pwd=len(valid.split("\n"))
    for i in range(no_valid_pwd):
        labels.append(1)
    labels = np.array(labels)

    train_data= []
    reader = valid.split("\n")
    print(reader)
    for password in reader:
        if ' ' in password:
            li=password.split(' ')
            password=li[0]+li[1]
        train_data.append(encodeString(password))
    train_data=np.array(train_data)
#    print(train_data)

    x_train = train_data
    train_size=x_train.shape[0]
    input_size=x_train.shape[1]
    steps_per_epoch = round(train_size/batch_size)
    
    discriminator = create_discriminator(input_size)
    generator = create_generator(input_size)
    discriminator.trainable = False

    gan_input = Input(shape=(noise_dim,))
    fake_image = generator(gan_input)
    gan_output = discriminator(fake_image)
    gan = Model(gan_input, gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=optimizergan)

    d_loss = 0
    g_loss = 0
    res = []
    str = ""
    for epoch in range(epochs):
        for batch in range(steps_per_epoch):
            noise = np.random.rand(batch_size, noise_dim)
            for z1 in range(batch_size):
                for z2 in range(noise_dim):
                    noise[z1][z2] *= np.random.choice([1,-1])
            fake_x = generator.predict_on_batch(noise)
            real_x = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]
            
            if epoch %1 ==0 :
                print(decodeList(fake_x[0]))
                print(decodeList(real_x[0]))
                print("\n")
            
            x = np.concatenate((real_x, fake_x))
            disc_y = np.zeros(2*batch_size)
            disc_y[:batch_size] = 0.9
            
            if epoch % 2 == 0:
                for i in range(2):
                    disc_y = np.reshape(disc_y, (2*batch_size,1))
                    d_loss = discriminator.train_on_batch(x, disc_y)
                
            y_gen = np.ones(batch_size)
            y_gen = np.reshape(y_gen,(batch_size,1))
            if epoch % 2 != 0:
                for i in range(1):
                    g_loss = gan.train_on_batch(noise, y_gen)
        if epoch%5==0:
            print(f'Epoch: {epoch} \t Discriminator Loss: {d_loss} \t\t Generator Loss: {g_loss}')
            str = f'Epoch: {epoch} \t Discriminator Loss: {d_loss} \t\t Generator Loss: {g_loss}'
            res.append(str)
    save_model(model_name, generator, discriminator, gan)
    str = "----------------------Model " + model_name + " saved to Disk----------------------"
    res.append(str)
    print(res)
    return res