### Import helper_functions
source("helper_functions.R")

### Install and import required packages
# We are using a helper function that takes the name of a package as argument
# Checks if it's installed and installs it if not
# Loads the package
# List of required packages
package_list = c("dplyr", "tidyr", "ggplot2", "scales", "mapproj")

# Call on convenience function to install and import packages
for (package in package_list) {
  usePackage(package)
}

### Set plotting parameters and import color palettes
set_ds_theme()

load("processed_uber_nyc.RData")
locID_zone_dim = locID_zone_dim = unique(agg_data[, c("locationID", "zone")])

### Precalculate mean pickups in each location by hour
agg_data$hour = as.numeric(agg_data$hour)
geo_pickups = agg_data %>%
  group_by(locationID, hour) %>%
  summarize(mean_pickup = mean(pickups))

# Define UI for application that draws a heatmap
ui = fluidPage(
  
  # Application title
  titlePanel("Pickups by Hour")
  
  # Sidebar with a slider input for number of bins 
  , sidebarLayout(
    sidebarPanel(
      sliderInput("hour",
                  "Hour of day:",
                  min = 0,
                  max = 23,
                  value = 24)
    )
    
    # Show a plot of the generated distribution
    , mainPanel(
      plotOutput("NYC_heatmap"
                 , height = 700
                 , width = 700
                 , hover = hoverOpts(id = "plot_hover"))
    )
  )
  # Tooltip
  , fluidRow(column(width = 6
                    , verbatimTextOutput("tooltip")))
)

# Define server logic required to draw a histogram
server = function(input, output) {
  ## Interactive Map ###########################################
  # Filter for data that we want
  hour_pickups = reactive({
    geo_pickups %>% 
      filter(hour == input$hour) %>%
      select(locationID, mean_pickup)
  })
  # Create the map
  heatmap = reactive({
    right_join(zone_polys, hour_pickups(), by = c("id" = "locationID"))
  })
  output$NYC_heatmap = renderPlot({
    ggplot() +
      geom_polygon(data = heatmap()
                   , aes(x = long, y = lat, group = id, fill = mean_pickup)
                   , color = "black"
                   , size = 0.25) +
      scale_fill_continuous(cont_gradient
                            , name = "Mean pickups by hour"
                            , breaks = pretty_breaks (n = 7)
                            , limits = c(0, 120)) +
      guides(fill = guide_legend(reverse = TRUE)) +
      xlim(-74.05, -73.8) +
      ylim(40.6, 40.85) +
      coord_map() +
      labs(x = "Latitude"
           , y = "Longitude"
           , title = paste0("Pickups By Zone at "
                            , input$hour)
      )
  })
  
  # Check which polygon the mouse is hovering over
  
  output$tooltip = renderPrint({
    paste0("Longitude: "
           , as.character(round(input$plot_hover$x, digits = 4))
           , "; "
           , "Latitude: "
           , as.character(round(input$plot_hover$y, digits = 4))
    )
  })
}

# Run the application 
shinyApp(ui = ui, server = server)