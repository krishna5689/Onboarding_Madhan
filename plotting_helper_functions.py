import featurize_helper_functions as fhf
import numpy as np
from plotly import tools
import plotly.graph_objs as go
from matplotlib import pyplot as plt

def get_job_groups():
    # The data contains many user jobs.  We'll bin them into similar groups
    job_groups = {
        ('administrator','executive','marketing','salesman') : 'Business / Management',
        ('librarian','educator') : 'Education',
        ('doctor', 'healthcare') : 'Healthcare',
        ('lawyer',) : 'Law',
        ('other', 'none', 'technician','homemaker') : 'Other',
        ('artist', 'writer', 'entertainment') : 'Content Producer',
        ('retired',): 'Retired',
        ('engineer', 'programmer','scientist') : 'Technical',
        ('student',) : 'Student'
    }
    
    return job_groups

def get_avg_hg_ratings(user_features, item_features):

    avg_ratings = []
    for gi, g in enumerate(item_features.columns):
        g_mid = item_features.index.values[np.where(item_features[g].values==1)[0]]
        g_idx = np.where(np.in1d(user_features.columns.values, g_mid.astype(str)))[0]
        avg_ratings.append(np.nanmean(user_features.iloc[:,g_idx],1))
    avg_ratings = np.array(avg_ratings)
    
    high_g_ratings = []
    for i in avg_ratings:
        high_g_ratings.append(np.where(i >= 0.8)[0])
    high_g_ratings = np.array(high_g_ratings)
    
    return avg_ratings, high_g_ratings

def user_scatter_plot(user_features):

    job_groups = get_job_groups()
    # Scatter latitude (x), longitude(y), and Age(z), and color code by occupation
    lat = np.array([fhf.get_zip_data(z,'latitude') for z in user_features.zip_code])
    lon = np.array([fhf.get_zip_data(z,'longitude') for z in user_features.zip_code])
    x = lat /90
    y = lon / 180
    z = np.array(user_features.normed_age*130) #Age was normalized by 130, un-normalize for plot

    # Generate a list of colors for plotting, one for each job group
    color_list = fhf.get_N_HexCol(len(job_groups.keys()), 0.6, 0.75)


    # Generate the employment bins, and for each one, generate its scatter plot data
    traces = []
    for ei, e in enumerate(job_groups.keys()):

        # Generate the employment bins
        e_idx = []
        for jobs in e:
            e_idx.extend(np.where(user_features[jobs]==1)[0].tolist())
        e_idx = np.array(e_idx)

        # Generate the scatter data for each bin, and append it to the 'traces' list
        # which we will then plot
        x_i = x[e_idx]
        y_i = y[e_idx]
        z_i = z[e_idx]
        traces.append(
            go.Scatter3d(x=x_i, y=y_i, z=z_i, name = job_groups[e], mode='markers',
                marker=dict(color = color_list[ei], size=5,
                            line=dict(color='rgba(58, 71, 80, 1.0)',width=0.5),
                            opacity=0.8)
            )
        )

    # Set up the 3d plot's camera position
    camera = dict(
        #up=dict(x=0, y=0, z=1),
        #center=dict(x=0, y=0, z=0),
        eye=dict(x=1.35, y=1.35, z=0.65)

    )

    # Set up the plot's layout
    layout = dict(
        #height = 650,
        height = 500,
        width = 780,
        #width = 900,
        scene = dict(camera=camera, xaxis=dict(title='Latitude',),
                     yaxis=dict(title='Longitude',),
                     zaxis=dict(title='Age',)),
        margin=dict(l=30, r=30, t=30, b=30,),

    )

    # Get plotting traces and layout
    return traces, layout
    
    

def user_job_stacked_bar(user_features):
    # Plot the percentage of users for each occupation

    # Generate occupation percentages
    job_groups = get_job_groups()
    job_perc = []
    jp_labels = []
    color_list = fhf.get_N_HexCol(len(job_groups.keys()), 0.6, 0.75)
    for ei, e in enumerate(job_groups.keys()):
        e_idx = []
        jp_labels.append(job_groups[e])
        for jobs in e:
            e_idx.extend(np.where(user_features[jobs]==1)[0].tolist())
        job_perc.append(np.float64(len(e_idx))/len(user_features))
    job_perc = np.round(np.array(job_perc)*100)
    jp_labels = np.array(jp_labels)
    sortidx = np.argsort(job_perc)


    # For each occupation, create a separate bar, and append it to trace1
    # These will be plotted as a stacked bar plot
    y_j = job_perc[sortidx].tolist()
    x_j = jp_labels[sortidx].tolist()
    trace1 = []
    for xi, x_ji in enumerate(x_j):
        trace1.append(go.Bar(y=['Job Percentages'], x=[y_j[xi]], name = x_ji, orientation = 'h',
            marker = dict(color = np.array(color_list)[sortidx].tolist()[xi], 
                          line = dict(color = 'rgba(58, 71, 80, 1.0)',width = 2)))
        )

    # Set the layout of the figure, and set barmode to 'stack' for stacked bar plot
    layout = dict(barmode = 'stack',showlegend = False, height=300,
                 margin=dict(l=100, r=20, t=170, b=70,))


    # Get plotting traces and layout
    return trace1, layout
    
    
def inventory_hbar(user_features, item_features):
    # Compute the average rating each user rates each movie genre


    # Get user's average ratings, and pull out the index of users that rated each genre 4/5 or higher
    avg_ratings, high_g_ratings = get_avg_hg_ratings(user_features, item_features)

    # Compute the percentage of users who rated each genre 4/5 or higher
    perc_users = []
    for hgi, hg in enumerate(high_g_ratings):
        perc_users.append(np.float64(len(hg))/np.sum(~np.isnan(avg_ratings[hgi])))

    perc_users = np.round(100*np.array(perc_users))

    # Compute the percentage of movies that make up each genre
    percentages = np.round((100*np.sum(item_features)/np.sum(np.sum(item_features))))
    
    sortidx = np.argsort(percentages)

    # Generate bar plot data for the percengate of items that make up each genre in the library
    y_pi = percentages[sortidx].tolist()
    x_i = item_features.columns[sortidx].tolist()
    trace0 = go.Bar(x=y_pi, y=x_i, marker=dict(color='rgba(171, 50, 96, 0.6)',line=dict(color='rgba(171, 50, 96, 1.0)',width=1),),
        name='Percentage of movies in genre',orientation='h')

    # Generate bar plot data for the percentage of users who rated each item highly
    y_pu = perc_users[sortidx].tolist()
    x_u = item_features.columns[sortidx].tolist()
    trace1 = go.Bar(x=y_pu, y=x_u, marker=dict(color='rgba(50, 171, 96, 0.6)',line=dict(color='rgba(50, 171, 96, 1.0)',width=1),),
        name='Percentage users rating genre high',orientation='h'
    )

    # Create the layout of the plot
    layout = dict(yaxis1=dict(showgrid=False,showline=True,linewidth=1,showticklabels=True,domain=[0, 0.85],),
                  yaxis2=dict(showgrid=False,showline=True,showticklabels=False,linewidth=1,domain=[0, 0.85],),
                  xaxis1=dict(zeroline=False,showline=False,showticklabels=True,showgrid=True,domain=[0, 0.42],),
                  xaxis2=dict(zeroline=False,showline=False,showticklabels=True,showgrid=True,domain=[0.47, 1],),
                  legend=dict(x=0.029,y=1.038,font=dict(size=10,),),
                  margin=dict(l=100,r=20,t=70,b=70),
                  paper_bgcolor='rgb(248, 248, 255)', plot_bgcolor='rgb(248, 248, 255)',
    )


    
    # Get plotting traces and layouts
    return trace0, trace1, layout

    
    
def conditional_interest(user_features, item_features):
    # Compute conditional ratings:
    # e.g. given that a user rated a given genre highly ...
    # ... how did the user rate each of the other genres?
    
    avg_ratings, high_g_ratings = get_avg_hg_ratings(user_features, item_features)
    cond_ratings = []
    for gi, g in enumerate(item_features.columns):
        cond_ratings.append([])
        g_mid = item_features.index.values[np.where(item_features[g].values==1)[0]]
        g_idx = np.where(np.in1d(user_features.columns.values, g_mid.astype(str)))[0]
        for gi2, g2 in enumerate(item_features.columns):
            g2_mid = item_features.index.values[np.where(item_features[g2].values==1)[0]]
            g2_idx = np.where(np.in1d(user_features.columns.values, g2_mid.astype(str)))[0]

            cond_ratings[gi].append(np.nanmean(user_features.iloc[high_g_ratings[gi],g2_idx]))

    cond_ratings = np.array(cond_ratings)

    # Normalize data within each row by the maximum value in that row
    cond_norm = cond_ratings/np.max(cond_ratings,0)

    # Make a heat map  / color matrix of the normalized ratings
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(cond_norm, cmap=plt.cm.Blues, alpha=0.8)

    # Set figure size
    fig = plt.gcf()
    fig.set_size_inches(12,6)

    # turn off the frame
    ax.set_frame_on(False)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(cond_norm.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(cond_norm.shape[1]) + 0.5, minor=False)

    # reverse the axes so they go from top to bottom, left to right
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top") 

    # Set the labels
    labels = item_features.columns.values
    ax.set_xticklabels(labels, minor=False)
    ax.set_yticklabels(labels, minor=False)

    # rotate the text of the x axis to be vertical
    plt.xticks(rotation=90)

    # turn off grid
    ax.grid(False)

    # Turn off all the ticks
    ax = plt.gca()
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # Set titles for axes
    plt.xlabel('... like [genre] this much.')
    plt.ylabel('People who like [genre]...')

    # Show plot
    plt.show();