from scripts.monk_script import monkTrain

if __name__ == '__main__':
    print("Monk #3")

    print("Batch (simple)")
    monkTrain(4,
              "datasets/monk3/monks-3.train",
              "datasets/monk3/monks-3.test",
              "plots/monk3/monk3_batch_noreg_{}.csv",
              200, 0.0, 0.2, 0.0, 1, "b", 0.6)

    print("Batch (reg)")
    monkTrain(4,
              "datasets/monk3/monks-3.train",
              "datasets/monk3/monks-3.test",
              "plots/monk3/monk3_batch_reg_{}.csv",
              500, 0.0, 0.4, 0.001, 1, "b", 0.6)

    print("Online")
    monkTrain(4,
              "datasets/monk3/monks-3.train",
              "datasets/monk3/monks-3.test",
              "plots/monk3/monk3_online_{}.csv",
              200, 0.0, 0.1, 0.0, 1, "o")