import os
import zipfile

_A1_FILES = [
    "pytorch101.py",
    "pytorch101.ipynb",
    "knn.py",
    "knn.ipynb",
]

_A2_FILES = [
    "linear_classifier.py",
    "linear_classifier.ipynb",
    "two_layer_net.py",
    "two_layer_net.ipynb",
    "svm_best_model.pt",
    "softmax_best_model.pt",
    "nn_best_model.pt",
]

_A3_FILES = [
    "fully_connected_networks.py",
    "fully_connected_networks.ipynb",
    "convolutional_networks.py",
    "convolutional_networks.ipynb",
    "best_overfit_five_layer_net.pth",
    "best_two_layer_net.pth",
    "one_minute_deepconvnet.pth",
    "overfit_deepconvnet.pth",
]

_A4_FILES = [
    "pytorch_autograd_and_nn.py",
    "pytorch_autograd_and_nn.ipynb",
    "two_stage_detector.py",
    "two_stage_detector.ipynb",
    "pytorch_autograd_and_nn.pt",
    "rcnn_detector.pt",
]


def make_a1_submission(assignment_path, name=None, idnum=None):
    _make_submission(assignment_path, _A1_FILES, "A1", name, idnum)


def make_a2_submission(assignment_path, name=None, idnum=None):
    _make_submission(assignment_path, _A2_FILES, "A2", name, idnum)


def make_a3_submission(assignment_path, name=None, idnum=None):
    _make_submission(assignment_path, _A3_FILES, "A3", name, idnum)


def make_a4_submission(assignment_path, name=None, idnum=None):
    _make_submission(assignment_path, _A4_FILES, "A4", name, idnum)


def _make_submission(
    assignment_path, file_list, assignment_no, name=None, idnum=None
):
    if name is None or idnum is None:
        name, idnum = _get_user_info(name, idnum)
    zip_path = "{}-{}-{}.zip".format(name.lower().replace(' ', '_'), idnum, assignment_no)
    zip_path = os.path.join(assignment_path, zip_path)
    print("Writing zip file to: ", zip_path)
    with zipfile.ZipFile(zip_path, "w") as zf:
        for filename in file_list:
            if filename.startswith('common/'):
                filename_out = filename.split('/')[-1]
            else:
                filename_out = filename
            in_path = os.path.join(assignment_path, filename)
            if not os.path.isfile(in_path):
                raise ValueError('Could not find file "%s"' % filename)
            zf.write(in_path, filename_out)


def _get_user_info(name, idnum):
    if name is None:
        name = input("Enter your name (e.g. kibok lee): ")
    if idnum is None:
        idnum = input("Enter your id number (e.g. 2022123456): ")
    return name, idnum
