import matplotlib.pyplot as plt
import numpy as np

num_imgs = 200
sigmas= np.array([0, 0.001, 0.01, 0.02, 0.03, 0.05, 0.07, 0.09, 0.11])
num_correct_clean=np.array([124., 124., 125., 123., 121., 121., 119., 119., 118.])
num_correct_rob=np.array([12., 13., 13., 14., 20., 36., 56., 78., 89.])
num_adv_success = np.array([177., 177., 176., 176., 164., 137.,  91.,  56.,  34.])


def plot_data(
        sigmas, num_imgs,
        num_adv_success_a1d0, num_correct_clean_d0,num_correct_rob_a1d0,
        num_adv_success_a1d1, num_correct_clean_d1, num_correct_rob_a1d1,
        num_adv_success_a2d1, num_correct_rob_a2d1,
        num_adv_success_a2d2, num_correct_clean_d2, num_correct_rob_a2d2,
        log=False,
):
    if log:
        xlabelstr=' [log scale]'
    else:
        xlabelstr=' [-]'

    fig, axes = plt.subplots(2,2)
    fig.suptitle('200 images')

    ax = axes[0,0]
    ax.plot(sigmas,num_correct_clean_d0/num_imgs, label = 'clean acc (A0,D0)')
    ax.plot(sigmas,num_correct_rob_a1d0/num_imgs, label = 'robust acc (A1,D0)')
    ax.plot(sigmas,num_adv_success_a1d0/num_imgs, label = 'adv success rate (A1,D0)')
    # ax[0].plot(sigmas,num_correct_clean/num_imgs, label = 'baseline clean acc (A0,D0)')
    ax.set_ylabel('acc/success rate [-]')
    ax.set_xlabel('sigma'+xlabelstr)
    ax.legend()
    ax.set_title('(A1,D0)')

    ax = axes[0,1]
    ax.plot(sigmas,num_correct_clean_d1/num_imgs, label = 'clean acc (A0,D1)')
    ax.plot(sigmas,num_correct_rob_a1d1/num_imgs, label = 'robust acc (A1,D1)')
    ax.plot(sigmas,num_adv_success_a1d1/num_imgs, label = 'adv success rate (A1,D1)')
    ax.plot(sigmas,num_correct_clean_d0/num_imgs, 'k:', label = 'baseline: clean acc (A0,D0)')
    ax.set_ylabel('acc/success rate [-]')
    ax.set_xlabel('sigma'+xlabelstr)
    ax.legend()
    ax.set_title('(A1,D1)')

    ax = axes[1,0]
    ax.plot(sigmas,num_correct_clean_d1/num_imgs, label = 'clean acc (A0,D1)')
    ax.plot(sigmas,num_correct_rob_a2d1/num_imgs, label = 'robust acc (A2,D1)')
    ax.plot(sigmas,num_adv_success_a2d1/num_imgs, label = 'adv success rate (A2,D1)')
    ax.plot(sigmas,num_correct_clean_d0/num_imgs, 'k:',label = 'baseline: clean acc (A0,D0)')
    ax.set_ylabel('acc/success rate [-]')
    ax.set_xlabel('sigma'+xlabelstr)
    ax.legend()
    ax.set_title('(A2,D1)')

    ax = axes[1,1]
    ax.plot(sigmas,num_correct_clean_d2/num_imgs, label = 'clean acc (A0,D2)')
    ax.plot(sigmas,num_correct_rob_a2d2/num_imgs, label = 'robust acc (A2,D2)')
    ax.plot(sigmas,num_adv_success_a2d2/num_imgs, label = 'adv success rate (A2,D2)')
    ax.plot(sigmas,num_correct_clean_d0/num_imgs, 'k:', label = 'baseline: clean acc (A0,D0)')
    ax.set_ylabel('acc/success rate [-]')
    ax.set_xlabel('sigma_max'+xlabelstr)
    ax.legend()
    ax.set_title('(A2,D2)')

    plt.show()
    return
