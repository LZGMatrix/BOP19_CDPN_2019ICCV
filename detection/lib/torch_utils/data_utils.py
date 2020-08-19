def collate_fn(batch):
    """
    example:
        [(img1, target11, target12), (img2, target21, target22)] -->
        [(img1, img2), (target11, target12), (target21, target22)]
    Might need to stack by hand later.
    if target is stored in dict:
        targets = [{k:v.to(device) for k,v in t.items()} for t in targets]
    """
    return tuple(zip(*batch))
