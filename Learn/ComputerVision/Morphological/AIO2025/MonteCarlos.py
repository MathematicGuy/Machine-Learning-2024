import random, math


def MonteCarlos(Ns):
    Nc = 0
    s = 2 # square edge length
    r = 1

    for i in range(Ns):
        #? generate random point within r radius  
        x = random.random()*2 - 1 # [0, 1]*2 - 1 to keep value in [-1, 1] range
        y = random.random()*2 - 1
        
        #? check if random point within the circle radius of 1
        x2 = x**2
        y2 = y**2
        
        if math.sqrt(x2 + y2) <= r:
            Nc = Nc + 1
            
    pi = (s**2/r**2) * (Nc / Ns)

    return pi


if __name__ == '__main__':
    Ns = 2_000_000
    
    pi = MonteCarlos(Ns)
    print(pi)
    
        
    
