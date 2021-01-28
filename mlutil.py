# Collection of Useful Machine Learning workflow methods, and viz tools.
# such as grid search, full viz,specific plotting mechanisms etc.
#

# For Grid Search
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# For plotting
import matplotlib.pyplot as plt


def hyp_pipeline(self, X_train, X_test, y_train, y_test,
                       model, param_grid, cv=10, scoring_fit='neg_mean_squared_error',
                       do_probabilities=False, search_mode='GridSearchCV', n_iterations=0, n_jobs=-1):
        '''Hyper Parameter searching with GridSearch and RandomSearch'''

        fitted_model = None
    
        if(search_mode == 'GridSearchCV'):
            gs = GridSearchCV(
                estimator=model,
                param_grid=param_grid, 
                cv=cv, 
                n_jobs=n_jobs, 
                scoring=scoring_fit,
                verbose=1)
            fitted_model = gs.fit(X_train, y_train)
        elif(search_mode == 'RandomizedSearchCV'):
            rs = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid, 
                cv=cv,
                n_iter=n_iterations,
                n_jobs=n_jobs, 
                scoring=scoring_fit,
                verbose=1)
            fitted_model = rs.fit(X_train, y_train)
    
    
        if(fitted_model != None):
            if do_probabilities:
                pred = fitted_model.predict_proba(X_test)
            else:
                pred = fitted_model.predict(X_test)
            
        return fitted_model, pred

def plot_history(self,history):
        ''' Plot accuracy and loss of keras Models '''
        loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
        val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
        acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
        val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    
        if len(loss_list) == 0:
            print('Loss is missing in history')
            return 
    
         # As loss always exists
        epochs = range(1,len(history.history[loss_list[0]]) + 1)
    
    # Loss
        plt.figure(1)
        for l in loss_list:
            plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
        for l in val_loss_list:
            plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
    
    # Accuracy
        plt.figure(2)
        for l in acc_list:
            plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
        for l in val_acc_list:    
            plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()


# Utility function to train the Pytorch model


# model = torch.nn.Linear(3, 2)
# loss_fn = torch.nn.functional.mse_loss
# opt = torch.optim.SGD(model.parameters(), lr=1e-5)

# from torch.utils.data import DataLoader,TensorDataSet
# train_dl= DataLoader(TensorDataset(inputs, targets),batch_size,shuffle=True)

def pytorch_fit(num_epochs, model, loss_fn, opt, train_dl):
    
    # Repeat for given number of epochs
    for epoch in range(num_epochs):
        
        # Train with batches of data
        for xb,yb in train_dl:
            
            # 1. Generate predictions
            pred = model(xb)
            
            # 2. Calculate loss
            loss = loss_fn(pred, yb)
            
            # 3. Compute gradients
            loss.backward()
            
            # 4. Update parameters using gradients
            opt.step()
            
            # 5. Reset the gradients to zero
            opt.zero_grad()
        
        # Print the progress
        if (epoch+1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))


# Viz NN models (Keras Models)

from keras.utils import plot_model

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

# pass keras model  
def view_model(model,show_info = False):
    if show_info:
        return plot_model(model, to_file='model.png',show_shapes=True, show_layer_names=True)
    else:
        return SVG(model_to_dot(model).create(prog='dot', format='svg'))


# model = Sequential()
# model.add(Dense(2, input_dim=1, activation='relu'))
# model.add(Dropout(0.4))

# model.add(Dense(1, activation='sigmoid'))
# model.add(Dropout(0.4))

# model.compile(optimizer = 'adam', 
#               loss='binary_crossentropy', metrics=['accuracy'])

# view_model(model,show_info=True)


# Function: print_quantile_info(qu_dataset, qu_field)
#   Print out the following information about the data
#   - interquartile range
#   - upper_inner_fence
#   - lower_inner_fence
#   - upper_outer_fence
#   - lower_outer_fence
#   - percentage of records out of inner fences
#   - percentage of records out of outer fences
# Input: 
#   - pandas dataframe (qu_dataset)
#   - name of the column to analyze (qu_field)
# Output:
#   None

def print_quantile_info(qu_dataset, qu_field):
    a = qu_dataset[qu_field].describe()
    
    iqr = a["75%"] - a["25%"]
    print("interquartile range:", iqr)
    
    upper_inner_fence = a["75%"] + 1.5 * iqr
    lower_inner_fence = a["25%"] - 1.5 * iqr
    print("upper_inner_fence:", upper_inner_fence)
    print("lower_inner_fence:", lower_inner_fence)
    
    upper_outer_fence = a["75%"] + 3 * iqr
    lower_outer_fence = a["25%"] - 3 * iqr
    print("upper_outer_fence:", upper_outer_fence)
    print("lower_outer_fence:", lower_outer_fence)
    
    count_over_upper = len(qu_dataset[qu_dataset[qu_field]>upper_inner_fence])
    count_under_lower = len(qu_dataset[qu_dataset[qu_field]<lower_inner_fence])
    percentage = 100 * (count_under_lower + count_over_upper) / a["count"]
    print("percentage of records out of inner fences: %.2f"% (percentage))
    
    count_over_upper = len(qu_dataset[qu_dataset[qu_field]>upper_outer_fence])
    count_under_lower = len(qu_dataset[qu_dataset[qu_field]<lower_outer_fence])
    percentage = 100 * (count_under_lower + count_over_upper) / a["count"]
    print("percentage of records out of outer fences: %.2f"% (percentage))


# Function: remove_outliers_using_quantiles(qu_dataset, qu_field, qu_fence)
#   1- Remove outliers according to the given fence value and return new dataframe.
#   2- Print out the following information about the data
#      - interquartile range
#      - upper_inner_fence
#      - lower_inner_fence
#      - upper_outer_fence
#      - lower_outer_fence
#      - percentage of records out of inner fences
#      - percentage of records out of outer fences
# Input: 
#   - pandas dataframe (qu_dataset)
#   - name of the column to analyze (qu_field)
#   - inner (1.5*iqr) or outer (3.0*iqr) (qu_fence) values: "inner" or "outer"
# Output:
#   - new pandas dataframe (output_dataset)

def remove_outliers_using_quantiles(qu_dataset, qu_field, qu_fence):
    a = qu_dataset[qu_field].describe()
    
    iqr = a["75%"] - a["25%"]
    print("interquartile range:", iqr)
    
    upper_inner_fence = a["75%"] + 1.5 * iqr
    lower_inner_fence = a["25%"] - 1.5 * iqr
    print("upper_inner_fence:", upper_inner_fence)
    print("lower_inner_fence:", lower_inner_fence)
    
    upper_outer_fence = a["75%"] + 3 * iqr
    lower_outer_fence = a["25%"] - 3 * iqr
    print("upper_outer_fence:", upper_outer_fence)
    print("lower_outer_fence:", lower_outer_fence)
    
    count_over_upper = len(qu_dataset[qu_dataset[qu_field]>upper_inner_fence])
    count_under_lower = len(qu_dataset[qu_dataset[qu_field]<lower_inner_fence])
    percentage = 100 * (count_under_lower + count_over_upper) / a["count"]
    print("percentage of records out of inner fences: %.2f"% (percentage))
    
    count_over_upper = len(qu_dataset[qu_dataset[qu_field]>upper_outer_fence])
    count_under_lower = len(qu_dataset[qu_dataset[qu_field]<lower_outer_fence])
    percentage = 100 * (count_under_lower + count_over_upper) / a["count"]
    print("percentage of records out of outer fences: %.2f"% (percentage))
    
    if qu_fence == "inner":
        output_dataset = qu_dataset[qu_dataset[qu_field]<=upper_inner_fence]
        output_dataset = output_dataset[output_dataset[qu_field]>=lower_inner_fence]
    elif qu_fence == "outer":
        output_dataset = qu_dataset[qu_dataset[qu_field]<=upper_outer_fence]
        output_dataset = output_dataset[output_dataset[qu_field]>=lower_outer_fence]
    else:
        output_dataset = qu_dataset
    
    print("length of input dataframe:", len(qu_dataset))
    print("length of new dataframe after outlier removal:", len(output_dataset))
    
    return output_dataset
  
  #data is pandas df
  def replace_outliers(data):
    for col in data.columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        median_ = data[col].median()
        # data[col].mask(((data[col] < Q1 - 1.5*IQR) | (data[col] > Q3 + 1.5*IQR)), median_, inplace=True)
        # data[col] = np.where(((data[col] < Q1 - 1.5*IQR) | (data[col] > Q3 + 1.5*IQR)),
        #                     median_, data[col])
        data.loc[((data[col] < Q1 - 1.5*IQR) | (data[col] > Q3 + 1.5*IQR)), col] = median_
    return data
  
  
  
  
  
