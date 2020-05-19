from Emotion_recognition_CNN.Preprocessing import Preprocessing
from Emotion_recognition_CNN.Model import Model
from Emotion_recognition_CNN.Parameters import Parameters

import seaborn as sns
from matplotlib import pyplot
import sys, getopt


# function to input argument "path"
def command(argv):
    if len(argv) != 2:
        print("Wrong number of arguments.   Type main -h for help")
        sys.exit()
    try:
        opts, args = getopt.getopt(argv,"hi:",["ipath="])
    except getopt.GetoptError:
        print ('main.py -i <path to dataset> ')
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print ('main.py -i <path to dataset> '
                'eg: <fer2013/fer2013.csv>')
            sys.exit()
        elif opt in ("-i", "--ipath"):
            inputfile = arg
            return inputfile


if __name__ == "__main__":

    # setup path from input argument
    path = command(sys.argv[1:])

    # initiate the preprocessing
    pre = Preprocessing(path)
    img_width, img_height, img_depth, num_classes, X_train, X_valid, y_train, y_valid = pre.preprocessing()

    # initiate the model parameters
    para = Parameters(X_train)
    callbacks, train_datagen, batch_size, epochs, optims = para.parameters()

    # run the model
    mod = Model(img_width, img_height, img_depth, num_classes, X_train, X_valid, optims)
    model = mod.build_net()
    # based on the model and the parameters, run it
    history = model.fit_generator(
        train_datagen.flow(X_train, y_train, batch_size=batch_size),
        validation_data=(X_valid, y_valid),
        steps_per_epoch=len(X_train) / batch_size,
        epochs=epochs,
        callbacks=callbacks,
        use_multiprocessing=True
    )

    # save model architecture to yaml file
    model_yaml = model.to_yaml()
    with open("model.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    # save the trained model to h5 file
    model.save("model.h5")

    # plot the results of accuracy and loss over epoch
    sns.set()
    fig = pyplot.figure(0, (12, 4))

    ax = pyplot.subplot(1, 2, 1)
    sns.lineplot(history.epoch, history.history['accuracy'], label='train')
    sns.lineplot(history.epoch, history.history['val_accuracy'], label='valid')
    pyplot.title('Accuracy')
    pyplot.tight_layout()

    bx = pyplot.subplot(1, 2, 2)
    sns.lineplot(history.epoch, history.history['loss'], label='train')
    sns.lineplot(history.epoch, history.history['val_loss'], label='valid')
    pyplot.title('Loss')
    pyplot.tight_layout()

    # save the plots
    pyplot.savefig('model.png')
    pyplot.show()

    ##
    print("Runnable    -- end --")
    ##





