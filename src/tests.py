from Tester import Tester



tester = Tester('test.csv', 'my_model\checkpoint-36970')

print(tester.summary_metrics(500))
print(tester.avg_speed(500))