class ServerReply:
    def __init__(self, layer_grad, loss_message):
        self.layer_grad = layer_grad
        self.loss_message = loss_message

    def get_layer_grad(self):
        return self.layer_grad

    def get_loss_message(self):
        return self.loss_message