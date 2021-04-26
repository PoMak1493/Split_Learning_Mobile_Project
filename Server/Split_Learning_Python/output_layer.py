class output_layer:
    def __init__(self, outputs_layer, labels, layer_no):
        self.outputs_layer = outputs_layer
        self.labels = labels
        self.layer_no = layer_no

    def get_output_layer(self):
        return self.outputs_layer

    def get_labels(self):
        return self.labels

    def get_layer_no(self):
        return self.layer_no