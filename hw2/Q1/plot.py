from matplotlib import pyplot as plt

a = [-94.13550131490025, -195.98126186738796, -33.72930040399028, -73.59677803919651, -8.47154538879968, -2.320080843336547, -29.7234817874973, 1157.5855356141885, -5.564556201017805, -3.5520720878405774, -3.212647026585895, -2.1481170311835935, 1015.5979735616053]
plt.plot(a)
plt.xlabel('motif')
plt.ylabel('z score')
plt.title('USPowergrid')
plt.savefig('q3_3.png', format='png')
plt.show()