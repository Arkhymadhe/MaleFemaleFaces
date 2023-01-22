import os

import torch
from flask import render_template, Flask, request, jsonify

from viz_utils import display_images


HOME_TEMPLATE_FPATH = os.getcwd().replace("scripts", 'templates')
app = Flask(__name__, template_folder=HOME_TEMPLATE_FPATH)

@app.route("/")
def load_home():
    return render_template("homepage.html")


def load_model(PATH):
    loaded_model = torch.load(PATH)
    return loaded_model


def generate_seed(cond = 0, label = None, num_samples=1, latent_dims=(100,1,1)):
    seed = torch.randn(size=[num_samples, *latent_dims])

    if cond:
        labels = torch.randint(low=0, high=2, size=(num_samples,))
        if label:
            labels.fill_(label)
        seed = (seed, labels)

    return seed


def generate_imgs(model, seed):
    generated_images = model(seed)
    return generated_images if type(seed) != tuple else (generated_images, seed[1])


if __name__ == "__main__":
    app.run()