# class to manage the charts displayed in the application
class ChartsController:
    def __init__(self, variables, decomposition_thr):
        # flag to indicate if new data is available
        self.new_data = False
        # flag to force the update of the charts
        self.force_update = False
        # stores the data to be displayed in the charts
        self.decomposition, self.position = [], []
        self.decomposition_x = []
        self.decomposition_counter = 0
        self.variables = variables
        self.decomposition_thr = decomposition_thr
        self.window_size = 500

    def update_charts(self, value):
        self.new_data = value

    def can_update(self):
        return self.new_data or self.force_update

    def map_position_insert(self, data):
        # self.position.append([data["X"], data["Y"], data["anomaly"]])
        # apply window shift
        if len(self.position) > self.window_size:
            self.position = self.position[1:]
            self.position.append([data["X"], data["Y"], data["anomaly"]])
        else:
            self.position.append([data["X"], data["Y"], data["anomaly"]])
        # set new data to true, to trigger the update of the charts
        self.new_data = True

    def variable_decomposition_insert(self, data):
        # self.decomposition.append([data[var] for var in self.variables])
        # apply window shift
        if len(self.decomposition) > self.window_size:
            self.decomposition = self.decomposition[1:]
            self.decomposition.append([data[var] for var in self.variables])
            self.decomposition_x = self.decomposition_x[1:]
            self.decomposition_x.append(self.decomposition_counter)
        else:
            self.decomposition.append([data[var] for var in self.variables])
            self.decomposition_x.append(self.decomposition_counter)
        # set new data to true, to trigger the update of the charts
        self.decomposition_counter += 1
        self.new_data = True

    def reset(self):
        self.decomposition, self.position = [], []
        self.decomposition_x = []
        self.decomposition_counter = 0
        self.new_data = True
