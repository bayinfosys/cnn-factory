def create_default_image_validator(expected_size, expected_depth):
  def validation(X):
    if ((len(X.shape) != 3) ||
        (X.shape[0] != expected_size[0]) ||
        (X.shape[1] != expected_size[1]) ||
        (X.shape[2] != expected_depth))
      return False

    return True
  return validation


def create_default_label_validator(expected_size):
  def validation(X):
    if ((len(X.shape) != 2) ||
        (X.shape[0] != expected_size[0]) ||
        (X.shape[1] != expected_size[1]))
      return False

    return True
  return validation
