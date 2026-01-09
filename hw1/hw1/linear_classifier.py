import torch
from torch import Tensor
from collections import namedtuple
from torch.utils.data import DataLoader

from .losses import ClassifierLoss
from .transforms import BiasTrick


class LinearClassifier(object):
    def __init__(self, n_features, n_classes, weight_std=0.001):
        """
        Initializes the linear classifier.
        :param n_features: Number of features in each sample.
        :param n_classes: Number of classes samples can belong to.
        :param weight_std: Standard deviation of initial weights.
        """
        self.n_features = n_features
        self.n_classes = n_classes

        # TODO:
        #  Create weights tensor of appropriate dimensions
        #  Initialize it from a normal dist with zero mean and the given std.

        self.weights = None
        # ====== YOUR CODE: ======
        weights_shape = (n_features, n_classes) #check correctness
        self.weights = (torch.randn(weights_shape) * weight_std)
        # The shape of the PyTorch weights is: {self.weights.shape}")
        # ========================

    def predict(self, x: Tensor):
        """
        Predict the class of a batch of samples based on the current weights.
        :param x: A tensor of shape (N,n_features) where N is the batch size.
        :return:
            y_pred: Tensor of shape (N,) where each entry is the predicted
                class of the corresponding sample. Predictions are integers in
                range [0, n_classes-1].
            class_scores: Tensor of shape (N,n_classes) with the class score
                per sample.
        """

        # TODO:
        #  Implement linear prediction.
        #  Calculate the score for each class using the weights and
        #  return the class y_pred with the highest score.

        y_pred, class_scores = None, None
        # ====== YOUR CODE: ======

        #print(f"The shape of the x is: {x.shape}")

        bias_trick_transform = BiasTrick()
        x_augmented = bias_trick_transform(x)

        #print(f"The shape of the x_augmented is: {x_augmented.shape}")


        class_scores = x @ self.weights

        y_pred = torch.argmax(class_scores, dim=1)
        # ========================

        return y_pred, class_scores

    @staticmethod
    def evaluate_accuracy(y: Tensor, y_pred: Tensor):
        """
        Calculates the prediction accuracy based on predicted and ground-truth
        labels.
        :param y: A tensor of shape (N,) containing ground truth class labels.
        :param y_pred: A tensor of shape (N,) containing predicted labels.
        :return: The accuracy in percent.
        """

        # TODO:
        #  calculate accuracy of prediction.
        #  Do not use an explicit loop.

        acc = None
        # ====== YOUR CODE: ======
        correct_predictions = (y == y_pred)
        correct_count = correct_predictions.float().sum()
        acc = correct_count / y.numel()
        # ========================

        return acc * 100

    def train(
        self,
        dl_train: DataLoader,
        dl_valid: DataLoader,
        loss_fn: ClassifierLoss,
        learn_rate=0.1,
        weight_decay=0.001,
        max_epochs=100,
    ):

        Result = namedtuple("Result", "accuracy loss")
        train_res = Result(accuracy=[], loss=[])
        valid_res = Result(accuracy=[], loss=[])

        print("Training", end="")


        for epoch_idx in range(max_epochs):
            total_correct = 0
            average_loss = 0

            total_loss = 0
            total_samples = 0
            # TODO:
            #  Implement model training loop.
            #  1. At each epoch, evaluate the model on the entire training set
            #     (batch by batch) and update the weights.
            #  2. Each epoch, also evaluate on the validation set.
            #  3. Accumulate average loss and total accuracy for both sets.
            #     The train/valid_res variables should hold the average loss
            #     and accuracy per epoch.
            #  4. Don't forget to add a regularization term to the loss,
            #     using the weight_decay parameter.

            # ====== YOUR CODE: ======
            for X, y in dl_train:
                #get batch size
                N_batch = X.shape[0]
                #eval loss
                y_pred, x_scores = self.predict(X)
                batch_loss = loss_fn.loss(X, y, x_scores, y_pred)
                reg_term = 0.5 * weight_decay * torch.sum(self.weights ** 2)
                #full_loss = batch_loss + reg_term

                #eval grad
                grad_weights = loss_fn.grad()
                grad_reg = weight_decay * self.weights
                total_grad = grad_weights + grad_reg

                #perform weight update
                self.weights -= learn_rate * total_grad
                #print(f"2 The shape of the PyTorch weights is: {self.weights.shape}")
                #accumulate batch results
                total_loss += batch_loss.item() * N_batch
                total_correct += torch.sum(y == y_pred).item()
                total_samples += N_batch

            #calc epoch results
            train_accuracy = (total_correct / total_samples) * 100
            average_loss = total_loss / total_samples

            #update epoch train results
            train_res.accuracy.append(train_accuracy)
            train_res.loss.append(average_loss)

            #calc epoch valid results
            valid_total_samples = 0
            valid_total_correct = 0
            valid_total_loss = 0

            for X, y in dl_valid:
                N_batch = X.shape[0]
                y_pred, x_scores = self.predict(X)

                batch_loss = loss_fn.loss(X, y, x_scores, y_pred)
                reg_term = 0.5 * weight_decay * torch.sum(self.weights ** 2) #make sure needed
                #full_loss = batch_loss + reg_term #make sure needed

                batch_correct = torch.sum(y == y_pred).item()
                valid_total_loss += batch_loss.item() * N_batch  # Multiply by batch size to get total loss, not avg loss
                valid_total_correct += batch_correct
                valid_total_samples += N_batch
            #update epoch valid results
            valid_accuracy = (valid_total_correct / valid_total_samples) * 100
            valid_avg_loss = valid_total_loss / valid_total_samples

            valid_res.accuracy.append(valid_accuracy)
            valid_res.loss.append(valid_avg_loss)
            # ========================
            print(".", end="")

        print("")
        return train_res, valid_res

    def weights_as_images(self, img_shape, has_bias=True):
        """
        Create tensor images from the weights, for visualization.
        :param img_shape: Shape of each tensor image to create, i.e. (C,H,W).
        :param has_bias: Whether the weights include a bias component
            (assumed to be the first feature).
        :return: Tensor of shape (n_classes, C, H, W).
        """

        # TODO:
        #  Convert the weights matrix into a tensor of images.
        #  The output shape should be (n_classes, C, H, W).

        # ====== YOUR CODE: ======
        # 1. Separate the weights from the optional bias term.
        print(f"The shape of the PyTorch weights is: {self.weights.shape}")

        if has_bias:
            # The weights tensor shape is (D+1, C). We discard the first row (the bias).
            # w_matrix_to_reshape has shape (D, C)
            w_matrix_to_reshape = self.weights[1:, :]
        else:
            # Weights tensor shape is (D, C).
            w_matrix_to_reshape = self.weights[:, :]

        print(f"The shape of the PyTorch tensor is: {w_matrix_to_reshape.shape}")
        # 2. Transpose the matrix to group weights by class.
        # Required shape for reshaping is (n_classes, D).
        # Original shape (D, C) is transposed to (C, D) = (n_classes, D).
        w_transposed = w_matrix_to_reshape.T


        # 3. Reshape the tensor into image format.
        # C_img, H_img, W_img are the components of img_shape.
        # The total number of features (D) must equal C_img * H_img * W_img.
        # Output shape: (n_classes, C_img, H_img, W_img)
        C_img, H_img, W_img = img_shape
        w_images = w_transposed.reshape(w_transposed.size()[0],C_img, H_img, W_img)
        # ========================

        return w_images


def hyperparams():
    hp = dict(weight_std=0.0, learn_rate=0.0, weight_decay=0.0)

    # TODO:
    #  Manually tune the hyperparameters to get the training accuracy test
    #  to pass.
    # ====== YOUR CODE: ======
    hp = dict(
        weight_std=1e-3,  # Small value for weight initialization
        learn_rate=1e-3,  # Moderate learning rate for stable convergence
        weight_decay=1e-5  # Small L2 regularization to prevent overfitting
    )
    # ========================

    return hp
