def distance(s_line,train_data,feature,intercept):
    print(train_data.shape)
    deno=math.sqrt(np.sum(np.square(s_line)))
    dist=(abs((np.matmul(train_data[:,0:feature],s_line)+intercept))/deno)
    return dist
def data_cre(traing_data_X=None,train_data_Y=None,minority_class=0,major=0.50,minor=0.50):
    train_data=np.copy(traing_data_X)
    #np.place(train_data[:, 768],train_data[:, 768]!=minority_class, [0])
    #np.place(train_data[:, 768],train_data[:, 768]==minority_class, [1])
    train_data_1= train_data[train_data_Y ==minority_class ]
    train_data_non_1=train_data[ train_data_Y!=minority_class ]
    length_1=train_data_1.shape[0]
    length_non_1=train_data_non_1.shape[0]
    feature=train_data_1.shape[1]
    center_1=train_data_1[:,:].mean(axis=0).reshape(1,feature)
    center_non_1=train_data_non_1[:,:].mean(axis=0).reshape(1,feature)
    line_1=np.array([center_1,center_non_1]).reshape(2,feature)
    Y_1=[1,0]
    li_1=svm.SVC(kernel='linear')
    li_1.fit(line_1,Y_1)
    d_1=distance(li_1.coef_[0].reshape(feature,1),train_data_1[:,0:feature],feature,li_1.intercept_)
    d_non_1=distance(li_1.coef_[0].reshape(feature,1),train_data_non_1[:,0:feature],feature,li_1.intercept_)
    sorted_indices_1=np.argsort(d_1,axis=0)
    train_data_1=train_data_1[sorted_indices_1]
    sorted_indices_non_1=np.argsort(d_non_1,axis=0)
    train_data_non_1=train_data_non_1[sorted_indices_non_1]
    l1=train_data_1.shape[0]
    ln1=train_data_non_1.shape[0]
    y=[minority_class]*(math.ceil(major*l1))+[1-minority_class]*math.ceil(minor*ln1)
    y=np.array(y)
    train_data_1=train_data_1.reshape(l1,feature)
    train_data_non_1=train_data_non_1.reshape(ln1,feature)
    f_t= np.append(train_data_1[0:math.ceil(major*l1),:], train_data_non_1[0:math.ceil(minor*ln1),:], axis=0)
    indices = np.arange(f_t.shape[0]) 
    np.random.shuffle(indices)
    f_t=f_t[indices]
    y=y[indices]
    indices = np.arange(f_t.shape[0]) 
    np.random.shuffle(indices)
    f_t=f_t[indices]
    y=y[indices]
    return f_t,y