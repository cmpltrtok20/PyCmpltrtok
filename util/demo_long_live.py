import time

if '__main__' == __name__:
    
    def main():
        n = 0
        while True:
            n += 1
            print('.', end='', flush=True)
            if n % 10 == 0:
                print(f'\r{n}', end='', flush=True)
            time.sleep(1.0)
        
    main()
    