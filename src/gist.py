import torch


def reverse_cumsum(x: torch.Tensor) -> torch.Tensor:
    """Cumulative sum from right to left.

    See https://github.com/pytorch/pytorch/issues/33520.

    Args:
        x: a tensor of shape (batch_size, seq_len)
    Returns:
        A tensor of shape (batch_size, seq_len) where each element is the sum of
        all elements to the right of it.
    """
    return x + torch.sum(x, dim=-1, keepdims=True) - torch.cumsum(x, dim=-1)


def make_mask_pre_first_gist(
    inputs: torch.Tensor,
    gist_token: int,
    dtype=torch.int64,
) -> torch.Tensor:
    """Returns a mask where all tokens prior to the first gist token are masked out.
    Args:
        inputs: an array of input tokens where the last dimension is the
            sequence length.
        gist_token: the integer id of the gist token.
        dtype: the dtype of the mask, default int64.
    Returns:
        The requested mask.
    """
    return ((inputs == gist_token).cumsum(-1) >= 1).type(dtype)


def make_mask_post_last_gist(
    inputs: torch.Tensor,
    gist_token: int,
    dtype=torch.int64,
) -> torch.Tensor:
    """Returns a mask where all tokens after the last gist token are masked out.
    Computes the same as mask_pre_first_gist_token but reverses the
    sequence before and after the cumsum.
    Args:
        inputs: an array of input tokens where the last dimension is the
            sequence length.
        gist_token: the integer id of the gist token.
        dtype: the dtype of the mask, default int64.
    Returns:
        The requested mask.
    """
    return (reverse_cumsum(inputs == gist_token) >= 1).type(dtype)


def make_gist_mask(
    inputs: torch.Tensor, gist_token: int, dtype=torch.int64
) -> torch.Tensor:
    """Creates a 4D gist mask.
    Here, tokens after the last gist cannot attend to tokens prior to the first
    gist.
    Additionally, tokens *before* the last gist cannot attend to tokens *after*
    the last gist.

    Example, where G is the gist token:

      a b c G d
    a 1 1 1 1 0
    b 1 1 1 1 0
    c 1 1 1 1 0
    G 1 1 1 1 0
    d 0 0 0 1 1

    Args:
        inputs: an array of shape (batch_size, seq_len) input tokens.
        gist_token: the integer id of the gist token.
        dtype: the dtype of the mask, default int64.
    Returns:
        The requested mask of shape (batch_size, 1, seq_len, seq_len)
    """
    # Attention mask for tokens before the last gist token.
    pre_gist_mask = make_mask_post_last_gist(inputs, gist_token, torch.bool)[
        :, None, None
    ]
    # Attention mask for tokens after the last gist token.
    post_gist_mask = make_mask_pre_first_gist(inputs, gist_token, torch.bool)[
        :, None, None
    ]
    # Construct time masks by permuting to time dimension.
    pre_gist_time_mask = pre_gist_mask.permute((0, 1, 3, 2))

    return torch.where(pre_gist_time_mask, pre_gist_mask, post_gist_mask).type(dtype)


def make_neg_control_mask(inputs: torch.Tensor, gist_token: int, dtype=torch.int64):
    """Creates a 4D neg control mask.
    Here, tokens after the last gist cannot attend to any gist tokens (or prior).

    Example, where G is the gist token:

      a b c G d
    a 1 1 1 1 0
    b 1 1 1 1 0
    c 1 1 1 1 0
    G 1 1 1 1 0
    d 0 0 0 0 1

    Args:
        inputs: an array of shape (batch_size, seq_len) input tokens.
        gist_token: the integer id of the gist token.
        dtype: the dtype of the mask, default int64.
    Returns:
        The requested mask of shape (batch_size, 1, seq_len, seq_len)
    """
    # Attention mask for tokens before the last gist token.
    pre_gist_mask = make_mask_post_last_gist(inputs, gist_token, torch.bool)[
        :, None, None
    ]
    # Attention mask for tokens after the last gist token. This creates a mask
    # that is zero for all tokens up to and including the last gist token.
    post_gist_mask = torch.logical_not(pre_gist_mask)
    # Construct time masks by permuting to time dimension.
    pre_gist_time_mask = pre_gist_mask.permute((0, 1, 3, 2))

    return torch.where(pre_gist_time_mask, pre_gist_mask, post_gist_mask).type(dtype)
