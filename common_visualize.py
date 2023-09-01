import matplotlib.pyplot as plt


def visualize_cv_data(x, y, spr, spc, trans=None, size=(16, 8, )):
    plt.figure(figsize=size)
    spn = 0
    for i in range(spr * spc):
        spn += 1
        plt.subplot(spr, spc, spn)
        plt.axis('off')
        plt.title(f'{i}: {y[i]}')
        img = x[i]
        if trans is not None:
            img = trans(img)
        plt.imshow(img)
    print('Check and close the plotting window to continue ...')
    plt.show()