#!/bin/python3
import time
from random import randint

# computer:  my_choose
# 0 = r      0 = r
# 1  = s       1 = s
# 2 = p      1> = p

# 1 = r
# 2 = p
# 3 = s
import cv2

from get_image import count


def rock_game(thresholded, segmented, img):
    winner_name = ''
    player = int(count(thresholded, segmented))
    computer = randint(0, 2)
    if player > 2:
        player = 2
    if computer != player:
        if computer == 0:
            computer = 'rock'
            if player == 2:
                player = 'paper'
                winner_name = "player"
                print("player win")
            elif player == 1:
                player = 'scissors'
                winner_name = "computer"
                print("computer win")
        if computer == 2:
            computer = 'paper'
            if player == 1:
                player = 'scissors'
                winner_name = "player"
                print("player win")
            elif player == 0:
                player = 'rock'
                winner_name = "computer"
                print("computer win")
        if computer == 1:
            computer = 'scissors'
            if player == 1:
                player = 'rock'
                winner_name = "player"
                print("player win")
            elif player == 2:
                player = 'paper'
                winner_name = "computer"
                print("computer win")
    else:
        if computer == 0:
            computer, player = 'rock', 'rock'
        elif computer == 1:
            computer, player = 'scissors', 'scissors'
        else:
            computer, player = 'paper', 'paper'
        winner_name = "TIE"
        print("player: {} VS computer: {}".format(player, computer))
    return winner_name, player, computer


