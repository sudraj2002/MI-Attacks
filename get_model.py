    

def get_logits(inputs, labels=None , *args, **kwargs):

        logits = model(inputs)
        return logits


def get_target_label(labels=None):

        target_labels = labels

        return target_labels