

def cycle_dataloader(dl):
    while True:
        for data in dl:
            # print(data.shape)
            yield data



