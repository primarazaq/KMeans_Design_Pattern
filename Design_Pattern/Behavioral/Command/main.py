from command import KMeansCommand

def main():
    kluster = int(input("Masukkan Jumlah Kluster yang diinginkan : "))
    command = KMeansCommand(kluster)
    command.execute()

if __name__ == "__main__":
    main()
