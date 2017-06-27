from scripts.monk_script import monkTrain

if __name__ == '__main__':
    print("Monk #1")

    print("Batch")
    monkTrain(3,
              "datasets/monk1/monks-1.train",
              "datasets/monk1/monks-1.test",
              "plots/monk1/monk1_batch_noreg_{}.csv",
              3500, 0.0, 0.8, 0.0, 1, "b", 0.1)

    print("Online")
    monkTrain(3,
               "datasets/monk1/monks-1.train",
               "datasets/monk1/monks-1.test",
               "plots/monk1/monk1_online_{}.csv",
               500, 0.0, 0.2, 0.0, 1, "o")
