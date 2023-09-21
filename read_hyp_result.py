import pickle
x_t1=pickle.load(open('B-Box-eval/hyp_search_t2.x.pkl','rb'))
y_t1=pickle.load(open('B-Box-eval/hyp_search_t2.y.pkl','rb'))

x_t1[:,2]=x_t1[:,2]*3.0
x_t1[:,0]=x_t1[:,0]*5.0
y_t1[:,0]=-y_t1[:,0]

print(x_t1)
print(y_t1)


