from scripts.monk_script import monkTrain

if __name__ == '__main__':
    print("Monk #2")

    print("Batch")
    monkTrain(2,
              "datasets/monk2/monks-2.train",
              "datasets/monk2/monks-2.test",
              "plots/monk2/monk2_batch_noreg_{}.csv",
              3000, 0.0, 0.8, 0.0, 1, "b", 0.9)

    print("Online")
    monkTrain(2,
               "datasets/monks-2.train",
               "datasets/monks-2.test",
               "plots/monk2_online_{}.csv",
               300, 0.0, 0.05, 0.0, 1, "o")