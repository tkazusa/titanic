# coding=utf-8

# write code...

from multiprocessing import Pool, cpu_count, current_process
import time


def slowf(x):
    print(current_process().name,
          ': This started at %s.' % time.ctime().split()[3])

    time.sleep(1)
    return x * x


if __name__ == '__main__':
    print('cpu : %d' % 1)
    st = time.time()
    print('answer : %d' % sum(map(slowf, range(10))))
    print('time : %.3f s' % (time.time() - st))

    print('\ncpu : %d' % cpu_count())
    st = time.time()
    p = Pool()
    print('answer : %d' % sum(p.map(slowf, range(10))))
    print('time : %.3f s' % (time.time() - st))