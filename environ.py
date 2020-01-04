import numpy
import enum

from actionTable import choList, hanList
# noinspection PyArgumentList
Winner = enum.Enum("Winner", "black white draw")

# noinspection PyArgumentList
Player = enum.Enum("Player", "black white")

KING=1; CHA=2; PO=3; MA=4; SANG=5; SA=6; ZOL=7
maap=[[0,-1,-1,-2], [0,-1,1,-2], [1,0,2,-1], [1,0,2,1], [0,1,1,2], [0,1,-1,2], [-1,0,-2,-1], [-1,0,-2,1]]
sangap=[[0,-1,-1,-2,-2,-3], [0,-1,1,-2,2,-3], [1,0,2,-1,3,-2], [1,0,2,1,3,2],
	[0,1,1,2,2,3], [0,1,-1,2,-2,3], [-1,0,-2,1,-3,2], [-1,0,-2,-1,-3,-2]]
INITIAL_STATE = ((2,0,0,6,0,6,0,0,2),(0,0,0,0,1,0,0,0,0),(0,3,0,0,0,0,0,3,0),(7,0,7,0,7,0,7,0,7),
    (0,0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0,0),(17,0,17,0,17,0,17,0,17),(0,13,0,0,0,0,0,13,0),
    (0,0,0,0,11,0,0,0,0),(12,0,0,16,0,16,0,0,12))
MAX_TURN = 202

class Environment(object):
  """The environment MuZero is interacting with."""
  def __init__(self):
      self.board = None
      self.turn = 0
      self.done = False
      self.winner = None  # type: Winner
      self.resigned = False

  def reset(self):
      self.board = [list(i) for i in INITIAL_STATE]
      self.turn = 0
      self.done = False
      self.winner = None
      self.resigned = False
      return self

  def player_turn(self):
      if self.turn % 2 == 0:
          return Player.white
      else:
          return Player.black

  def step(self, action):
      chList = choList if self.turn % 2 < 1 else hanList
      action = chList[action]
      if (action >= 10000):
          if self.board[9][1] < 1:
              if action == 10000:
                  self.board[9][1] = 14; self.board[9][2] = 15; self.board[9][6] = 14; self.board[9][7] = 15
              elif action == 10001:
                  self.board[9][1] = 15; self.board[9][2] = 14; self.board[9][6] = 15; self.board[9][7] = 14
              elif action == 10002:
                  self.board[9][1] = 14; self.board[9][2] = 15; self.board[9][6] = 15; self.board[9][7] = 14
              else:
                  self.board[9][1] = 15; self.board[9][2] = 14;  self.board[9][6] = 14; self.board[9][7] = 15
          else:
              if action == 10000:
                  self.board[0][1] = 5; self.board[0][2] = 4; self.board[0][6] = 5; self.board[0][7] = 4
              elif action == 10001:
                  self.board[0][1] = 4; self.board[0][2] = 5; self.board[0][6] = 4; self.board[0][7] = 5
              elif action == 10002:
                  self.board[0][1] = 4; self.board[0][2] = 5; self.board[0][6] = 5; self.board[0][7] = 4
              else:
                  self.board[0][1] = 5; self.board[0][2] = 4; self.board[0][6] = 4; self.board[0][7] = 5
      elif action > 0:
          spos = action // 100; tpos = action % 100
          y0 = spos // 9; x0 = spos % 9; y1 = tpos // 9; x1 = tpos % 9
          captured = self.board[y1][x1]
          piece = self.board[y0][x0]
          self.board[y1][x1] = piece
          self.board[y0][x0] = 0

          if captured % 10 == KING:
              self.done = True
              self.winner = Winner.black if captured<10 else Winner.white

      self.turn += 1

      if self.turn >= MAX_TURN-1:
          self.done = True
          if self.winner is None:
              self.winner = Winner.draw

      r = 0
      if self.done:
        if self.turn % 2 == 0:
          if self.winner == Winner.white:
            r = 1
          elif self.winner == Winner.black:
            r = -1
        else:
          if self.winner == Winner.black:
            r = 1
          elif self.winner == Winner.white:
            r = -1
        r *= 1-(MAX_TURN-self.turn)/1000

      return r

  def legal_actions(self):
    legal = []
    if self.turn<2:
        for i in range(4): legal.append(10000+i)
    else:
        for y in range(10):
            for x in range(9):
                ki=self.board[y][x]; alk=ki//10
                if ki>0 and alk == self.turn%2:
                    k=y//7*7
                    if ki%10== KING or ki%10== SA:
                        if(x==4 and (y==1 or y==8)):
                            for i in range(-1,2):
                                for j in range(3,6):
                                    a=self.board[i+y][j]
                                    if a==0 or a//10 != alk: legal.append((y*9+x)*100+(i+y)*9+j)
                        else:
                            for i in range(3):
                                for j in range(3,6):
                                    if(abs(i+k-y)+abs(j-x)<2 or (i==1 and j==4)):
                                        a=self.board[i+k][j]
                                        if(a==0 or a//10!=alk):	legal.append((y*9+x)*100+(i+k)*9+j)
                    elif ki%10 == CHA:
                        if(x==4 and (y==1 or y==8)):
                            for i in range(-1,2,2):
                                for j in range(3,6,2):
                                    a=self.board[i+y][j]
                                    if(a==0 or a//10!=alk): legal.append((y*9+x)*100+(i+y)*9+j)
                        elif((x==3 or x==5) and (y==k or y==k+2)):
                            a=self.board[k+1][4]
                            if(a==0 or a//10!=alk): legal.append((y*9+x)*100+(k+1)*9+4)
                            b=2*k+2-y; a=self.board[b][8-x]
                            if(self.board[k+1][4]==0 and (a==0 or a//10!=alk)): legal.append((y*9+x)*100+b*9+8-x)
                        for i in range(x+1,9):
                            a=self.board[y][i]
                            if(a==0 or a//10!=alk): legal.append((y*9+x)*100+y*9+i)
                            if(a>0): break
                        for i in range(x-1,-1,-1):
                            a=self.board[y][i]
                            if(a==0 or a//10!=alk): legal.append((y*9+x)*100+y*9+i)
                            if(a>0): break
                        for i in range(y+1,10):
                            a=self.board[i][x]
                            if(a==0 or a//10!=alk): legal.append((y*9+x)*100+i*9+x)
                            if(a>0): break
                        for i in range(y-1,-1,-1):
                            a=self.board[i][x]
                            if(a==0 or a//10!=alk): legal.append((y*9+x)*100+i*9+x)
                            if(a>0): break
                    elif ki%10 == PO:
                        for i in range(x+1,8):
                            if(self.board[y][i]>0):
                                if(self.board[y][i]%10!=PO):
                                    for j in range(i+1,9):
                                        a=self.board[y][j]
                                        if(a==0 or (a//10!=alk and a%10!=PO)): legal.append((y*9+x)*100+y*9+j)
                                        if(a>0): break
                                break
                        for i in range(x-1,0,-1):
                            if(self.board[y][i]>0):
                                if(self.board[y][i]%10!=PO):
                                    for j in range(i-1,-1,-1):
                                        a=self.board[y][j]
                                        if(a==0 or (a//10!=alk and a%10!=PO)): legal.append((y*9+x)*100+y*9+j)
                                        if(a>0): break
                                break
                        for i in range(y+1,9):
                            if(self.board[i][x]>0):
                                if(self.board[i][x]%10!=PO):
                                    for j in range(i+1,10):
                                        a=self.board[j][x]
                                        if(a==0 or (a//10!=alk and a%10!=PO)): legal.append((y*9+x)*100+j*9+x)
                                        if(a>0): break
                                break
                        for i in range(y-1,0,-1):
                            if(self.board[i][x]>0):
                                if(self.board[i][x]%10!=PO):
                                    for j in range(i-1,-1,-1):
                                        a=self.board[j][x]
                                        if(a==0 or (a//10!=alk and a%10!=PO)): legal.append((y*9+x)*100+j*9+x)
                                        if(a>0): break
                                break
                        if((x==3 or x==5) and (y==k or y==k+2)):
                            a=self.board[k+1][4]; c=2*k+2-y; b=self.board[c][8-x]
                            if(a>0 and a%10!=PO and (b==0 or (b//10!=alk and b%10!=PO))):
                                legal.append((y*9+x)*100+c*9+8-x)
                    elif ki%10 == MA:
                        for i in range(8):
                            x1=x+maap[i][2]; y1=y+maap[i][3]
                            if(y1>=0 and y1<10 and x1>=0 and x1<9):
                                a=self.board[y1][x1]
                                if((a==0 or a//10!=alk) and self.board[y+maap[i][1]][x+maap[i][0]]==0):
                                    legal.append((y*9+x)*100+y1*9+x1)
                    elif ki%10 == SANG:
                        for i in range(8):
                            x1=x+sangap[i][4]; y1=y+sangap[i][5]
                            if(y1>=0 and y1<10 and x1>=0 and x1<9):
                                a=self.board[y1][x1]
                                if((a==0 or a//10!=alk) and self.board[y+sangap[i][1]][x+sangap[i][0]]==0 and self.board[y+sangap[i][3]][x+sangap[i][2]]==0):
                                    legal.append((y*9+x)*100+y1*9+x1)
                    elif ki%10 == ZOL:
                        ad=-1 if alk==1 else 1
                        for i in range(-1,2,):
                            x1=x+i; y1=y+(ad if i==0 else 0)
                            if(x1>=0 and x1<9 and y1>=0 and y1<10):
                                a=self.board[y1][x1]
                                if(a==0 or a//10!=alk): legal.append((y*9+x)*100+y1*9+x1)
                        if(x==4 and y==k+1):
                            for i in range(-1,2,2):
                                b=y+ad; a=self.board[b][x+i]
                                if(a==0 or a//10!=alk):	legal.append((y*9+x)*100+b*9+x+i)
                        if((x==3 or x==5) and (y==2 or y==7)):
                            a=self.board[k+1][4]
                            if(a==0 or a//10!=alk): legal.append((y*9+x)*100+(k+1)*9+4)
        legal.append(0)

    chList = choList if self.turn%2<1 else hanList
    for idx, a in enumerate(legal):
        legal[idx] = chList.index(a)
    return legal

  def black_and_white_plane(self):
      board_plane = numpy.zeros((14,10,9))
      bf = self.turn%2<1
      for i in range(10):
          ri = i if bf else 9-i
          for j in range(9):
              a = self.board[i][j]
              if a > 0: board_plane[(a//10 if bf else 1-a//10)*7+a%10-1][ri][j] = 1

      return numpy.array(board_plane)

  @property
  def observation(self):
      return ''.join(''.join(x for x in y) for y in self.board)


