from flask import Flask, request, render_template, jsonify
import os
from module import *

app = Flask(__name__)
init()

@app.route('/')
def home():
    return render_template('home.html') 

@app.route('/generate_pwd')
def generate_pwd():
    return render_template('generate_passwords.html')

@app.route('/train_model')
def train_model():
    return render_template('train_model.html')

@app.route('/pwd_strength')
def pwd_strength():
    return render_template("pwd_strength.html")

@app.route('/check_strength')
def check_strength():
    return render_template("a.html")

#########################################################3

@app.route('/generate')
def get_new_passwords():
    filename = request.args.get("GAN_MODEL_NAME_FOR_GENERATION")
    n = request.args.get("n")
#    print(filename , n, sep=" | ")
    generated_pwd = generate_new_passwords(filename, n)
    print(generated_pwd)
    return jsonify(generated_pwd)

#########################################################################33

@app.route('/train')
def train_new_gan():   
    filename = request.args.get("INPUT_FILE_NAME")
    model_name = request.args.get("model_name")
    #print(filename, model_name)
    res = train(filename, model_name) 
    return jsonify(res)

###########################################################################33

@app.route('/evaluate')
def evaluate_password_strength():
    password = request.args.get("password")
    model = request.args.get("GAN_MODEL_NAME_FOR_EVALUATION")
    print(model, password,sep=" | ")
    password_strength = check_strength_password(model, password)
    #return check_strength_password(model, password)
    return password_strength
#############################################################################3


@app.route('/gan')
def list_all_gan_models():
    
    list_of_trained_gan = os.listdir("./trained_gan")
    
    return_string = ""
    for i in list_of_trained_gan:
        return_string += i.split(".")[0]
        return_string += "\n"
    
    return return_string

@app.route('/d')
def list_all_discriminator_models():
    
    list_of_trained_gan = os.listdir("./trained_d")
    
    return_string = ""
    for i in list_of_trained_gan:
        return_string += i.split(".")[0]
        return_string += "\n"
    
    return return_string

if __name__=='__main__':
    app.run(debug=True)
