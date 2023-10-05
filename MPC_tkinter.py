import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy import array as arr
from numpy import transpose as tr

import ipywidgets as widgets
from IPython.display import display

import tkinter as tk
from tkinter import ttk
mpl.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


mpl.rcParams['image.origin']='lower'
mpl.rcParams['image.aspect']='auto'
mpl.rcParams['figure.figsize']=(8,6)
mpl.rcParams['axes.grid']=True




def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


class ray:
    def __init__(self, x, y, thetax, thetay, w0, sigma=1, z=0, wl=1030, M2=1):
        '''
        Initialise le rayon à une position sur le miroir 1, avec un angle.
        On lui donne son waist au foyer, et la distance entre ce foyer et le miroir (z).
        /!\ : z=0 signifie que le foyer est sur le miroir ! 
        Le waist donné est celui non-linéaire, qu'on espère avoir en fonctionnement optimal de la cellule.
        '''
        self.x = x
        self.y = y
        self.thetax = thetax
        self.thetay = thetay
        self.wavelength = wl
        self.sigma = sigma
        self.M2 = M2
        self.l_arts = []
        # self.q = z + 1j*w0**2 * np.pi/self.wavelength
        self.qnl = z + 1j*w0**2 * np.pi/self.wavelength / np.sqrt(self.sigma)

    def propagation(self, l):
        self.x += l*self.thetax
        self.y += l*self.thetay
        # self.q += l
        self.qnl += l
    
    def mirror(self, Roc):
        self.thetax -= 2*self.x/Roc
        self.thetay -= 2*self.y/Roc
        # self.q = self.q/(1-2/Roc*self.q)
        self.qnl = self.qnl/(1-2/Roc*self.qnl)

    def lens(self, f):
        self.qnl = self.qnl/(1-self.qnl/f)

    def translate_to_polar(self):
        return cart2pol(self.x, self.y)

    def waist0(self):
        return(np.sqrt(np.imag(self.qnl)/np.pi*self.wavelength*np.sqrt(self.sigma)))

    def waist(self):
        return(1/np.sqrt(np.imag(-1/self.qnl) *np.pi/np.sqrt(self.sigma)/self.wavelength))



class MPC_tk:

    def __init__(self, entries, reflectivity=1, n0=1, n2=0, n2_outside=0, wl=1030):
        print('----- Init -----')
        self.entries = entries
        self.cbtn_vars = {}
        self.outputs = {}
        self.outputs_bis = {}

        self.init_constants(reflectivity=reflectivity, n0=n0, n2=n2, n2_outside=n2_outside, wl=wl)


    def init_constants(self, reflectivity=1, n0=1, n2=0, n2_outside=0, wl=1030e-9):
        print(f'--- Constants of the MPC ---')
        print(f'Reflectivity of the mirrors: {reflectivity}')
        print(f'Non-linear index gas: {n2}')
        print(f'Non-linear index air: {n2_outside}')
        print(f'Wavelength: {wl}')
        self.reflectivity = reflectivity
        self.n0 = n0
        self.n2 = n2
        self.n2_outside = n2_outside
        self.beam_wavelength = wl *1e-9

    
    def update_entries(self):
        print('----- Updating entries -----')

        # print(self.entries)
        self.N = int(self.entries['N'].value())
        self.length = self.entries['L'].value()
        self.radius = self.entries['Radius'].value()*1e-3
        self.roc = self.entries['RoC'].value()
        self.pressure = self.entries['Pressure'].value()
        self.distance_to_window = self.entries['d_window'].value()
        self.window_to_lens = self.entries['d_lens'].value()
        self.lens_focal_length = self.entries['f'].value()

        self.x_in = self.entries['x'].value() * 1e-3
        self.y_in = self.entries['y'].value() * 1e-3
        self.thetax_in = self.entries['thetax'].value() * 1e-3
        self.thetay_in = self.entries['thetay'].value() * 1e-3

        self.beam_energy = self.entries['Energy'].value() *1e-3
        self.beam_waist_size = self.entries['Waist_foc'].value() * 1e-6
        self.beam_waist_lens = self.entries['Waist_lens'].value() * 1e-6
        self.beam_focus_pos = self.length/2
        self.tau0 = self.entries['Duration'].value() *1e-15
        self.beam_M2 = self.entries['M2'].value()

        # print(self.cbtn_vars['NL_propag'].get())
        # print(self.cbtn_vars['Propag from outside'].get())


        self.tau_eff = self.tau0 * np.sqrt(np.pi/(np.log(8)))
        self.power = self.beam_energy / self.tau0
        self.Peff = self.beam_energy / self.tau_eff
        self.Pcrit = 3.77*self.beam_wavelength**2/(8*np.pi*self.pressure*self.n2)
        if self.cbtn_vars['NL_propag'].get():
            self.sigma = 1 - self.power/self.Pcrit /np.sqrt(2)
            self.sigma_outside = 1 - self.power * (8*np.pi*self.n2_outside) / (3.77*self.beam_wavelength**2) / np.sqrt(2)
        else:
            self.sigma = 1
            self.sigma_outside = 1
        if self.sigma <= 0:
            print('The peak power is over the critical power. \n Look at what you did wrong')
        
        try:
            self.affichage()
        except AttributeError:
            print('???')
            pass


    def optimize_waist(self):
        # self.beam_waist_size = np.sqrt(self.beam_wavelength*self.length/(2*np.pi)*np.sqrt(self.sigma*(2*self.roc/self.length-1)))
        self.update_entries()
        w = np.sqrt(self.beam_wavelength*self.length/(2*np.pi)*np.sqrt(self.sigma*(2*self.roc/self.length-1)))
        self.update_entry('Waist_foc', w*1e6)
        self.beam_waist_size = w
        # self.update_entries()
        self.affichage()
        # return(self.beam_waist_size)


    def optimize_angle(self, axis=0, mode='sum'):
        
        tab_incr = [1, 0.1, 0.01, 0.01]
        startVal = 0
        R_in = np.sqrt(self.x_in**2+self.y_in**2)
        for j in range(len(tab_incr)):
            incr = tab_incr[j]

            t = []
            tab_angles = np.linspace(startVal-incr, startVal+incr, 100)
            for i in range(len(tab_angles)):
                if axis==0:
                    self.thetax_in = tab_angles[i]
                elif axis==1:
                    self.thetay_in = tab_angles[i]
                l_Rmirror, *_ = self.propag_MPC()
                if mode=='sum':
                    t.append(np.sum(l_Rmirror))
                elif mode=='var':
                    t.append(np.sum(np.var(l_Rmirror-R_in)))

            i1 = np.argmin(t)
            startVal = tab_angles[i1]
        if axis==0:
            self.thetax_in = tab_angles[i1]
        elif axis==1:
            self.thetay_in = tab_angles[i1]

    def optimize_injection(self):
        self.optimize_angle(0, 'sum')
        self.optimize_angle(1, 'var')
        self.optimize_angle(0, 'var')

        self.update_entry('thetax', self.thetax_in*1e3)
        self.update_entry('thetay', self.thetay_in*1e3)
        self.affichage()


    def update_entry(self, key, value):
        print(f'Updating entry {key} to {value}')
        _ = self.entries[key].delete(0, tk.END)
        _ = self.entries[key].insert(0, value)

    def waists(self):
        Ray_test = ray(self.x_in, self.y_in, self.thetax_in, self.thetay_in, sigma=self.sigma, wl=self.beam_wavelength, z=self.beam_focus_pos, w0=self.beam_waist_size, M2 = self.beam_M2)
        
        waist_mirror = Ray_test.waist()
        # sigma = Ray_test.sigma
        Fl = 2*self.beam_energy / (waist_mirror**2 * np.pi)

        dict_waists = {'Waist on mirrors':int(Ray_test.waist()*1e6*self.beam_M2)}
        Ray_test.propagation(-self.length/2)
        dict_waists['Waist at focus'] = int(Ray_test.waist()*1e6*self.beam_M2)
        Ray_test.propagation(-self.distance_to_window)
        dict_waists['Waist on window'] = int(Ray_test.waist()*1e6*self.beam_M2)

        qnl = Ray_test.qnl 
        ql = 1 / (np.real(1/qnl) + 1j * np.imag(1/qnl)/np.sqrt(Ray_test.sigma))
        Ray_test.qnl = ql
        Ray_test.sigma = 1
        Ray_test.propagation(-self.window_to_lens)
        dict_waists['Waist on lens'] = int(Ray_test.waist()*1e6*self.beam_M2)
        # print(1/(np.real(1/Ray_test.qnl)))    #Radius of curvature of the wavefront at lens
        Ray_test.propagation(+self.distance_to_window+self.window_to_lens)
        dict_waists['Waist for alignment'] = int(Ray_test.waist()*1e6*self.beam_M2)
        dict_waists['Fluence on mirrors'] = int(Fl*1e4)
        dict_waists['Sigma'] = self.sigma

        return(dict_waists)

    def retropropag(self, w_lens):
        self.Ray = ray(self.x_in, self.y_in, self.thetax_in, self.thetay_in, sigma=self.sigma_outside, wl=self.beam_wavelength, z=self.beam_focus_pos, w0=self.beam_waist_size, M2 = self.beam_M2)

        ql = 1 / (-1/self.lens_focal_length - 1j*self.beam_wavelength/(np.pi*w_lens**2)*np.sqrt(self.sigma_outside))
        self.Ray.qnl = ql
        self.Ray.propagation(self.window_to_lens)
        ql = self.Ray.qnl
        self.Ray.sigma = self.sigma
        qnl = 1/(np.real(1/ql)+1j*np.imag(1/ql)*np.sqrt(self.sigma))
        self.Ray.qnl = qnl
        self.Ray.propagation(self.distance_to_window)
        waist_middle = self.Ray.waist()
        error_pos_focus = np.real(self.Ray.qnl)
        w0 = self.Ray.waist0()
        return(waist_middle, error_pos_focus, w0)

    def propag_MPC(self, amont=False):
        if not amont:   #Si la propagation vient de l'amont, on a déjà un objet Ray avec lequel bosser
            self.Ray = ray(self.x_in, self.y_in, self.thetax_in, self.thetay_in, sigma=self.sigma, wl=self.beam_wavelength, z=self.beam_focus_pos, w0=self.beam_waist_size, M2=self.beam_M2)
        energy = self.beam_energy
        self.Bint = 0
        self.BintViotti = 0

        self.Ray.propagation(-self.length)
        self.xentry = self.Ray.x
        self.yentry = self.Ray.y
        self.waist_entry = self.Ray.waist()
        
        self.Bint += self.B_integral(energy=energy)
        self.BintViotti += self.B_integral(energy=energy, Viotti=True)
        
        self.Ray = ray(self.x_in, self.y_in, self.thetax_in, self.thetay_in, sigma=self.sigma, wl=self.beam_wavelength, z=self.beam_focus_pos, w0=self.beam_waist_size, M2=self.beam_M2)
        self.Ray.mirror(self.roc)
        l_Rcell = []
        l_Rmirror = [self.Ray.translate_to_polar()[0]]
        l_phimirror = [self.Ray.translate_to_polar()[1]]
        l_waist_mirror = []
        l_Bint = []

        for i in range(self.N):

            l_waist_mirror.append(self.Ray.waist())
            self.Ray.propagation(self.length/2)     #Propagate to the middle of the cell
            l_Rcell.append(self.Ray.translate_to_polar()[0])
            self.Bint += self.B_integral(energy=energy)
            l_Bint.append(self.B_integral(energy=energy))
            self.BintViotti += self.B_integral(energy=energy, Viotti=True)

            self.Ray.propagation(self.length/2)
            r, phi = self.Ray.translate_to_polar()
            self.Ray.mirror(self.roc)
            l_Rmirror.append(r)
            l_phimirror.append(phi)
            energy *= self.reflectivity
            sigma_i = 1 - energy/self.tau0 / self.Pcrit / np.sqrt(2)
            self.Ray.sigma = sigma_i
        

        self.b = 7*np.pi**2 * self.n2 * self.pressure * self.power * self.N / (2 * self.beam_wavelength**2)
        return(arr(l_Rmirror), arr(l_phimirror), arr(l_waist_mirror), arr(l_Bint))

    def B_integral(self, energy, Viotti=0):
        if not Viotti:
            return 4*np.pi*self.n2*self.pressure/self.beam_wavelength**2 * energy/self.tau0 /np.sqrt(self.sigma*self.beam_M2) * np.arctan(np.sqrt(self.length/(2*self.roc-self.length)))
        else:
            return 2*np.pi**2 * self.n2 * self.pressure * energy/self.tau0 / self.beam_wavelength**2 #* np.arccos(1-self.length/self.roc)


    def init_axes(self, fig, axes):
        self.fig = fig
        self.axes = axes
        self.circles_list = []
        for ax in self.axes:
            ax.set_xlim([-self.radius*1.1e3, self.radius*1.1e3])
            ax.set_ylim([-self.radius*1.1e3, self.radius*1.1e3])
            ax.add_artist(plt.Circle((0,0),self.radius*1e3, fill=False))


        # for i in range(self.N):     #Tentative pour ne pas avoir à recréer des cercles tout le temps, mais c'est pas plus rapide j'ai l'impression...
        #     # temp = self.axes[i%2].add_artist(plt.Circle(xy = [0,0], radius=1, color='k'))
        #     temp = plt.Circle(xy = [0,0], radius=1, color='k')
        #     self.axes[i%2].add_artist(temp)

        #     self.circles_list.append(temp)

    def affichage(self):

        if self.cbtn_vars['Propag from outside'].get():
            waist_middle, error_pos_focus, w0_bis = self.retropropag(w_lens=self.beam_waist_lens)
            print(f'Waist at the center of the MPC = {waist_middle*1e6} µm')
            print(f'Waist is off by = {error_pos_focus*1e3}mm')

            self.update_entry('Waist_foc', waist_middle*1e6)
            l_Rs, l_thetas, l_waists, l_Bints = self.propag_MPC(amont=True)
            dict_waists = self.waists()

        else:
            l_Rs, l_thetas, l_waists, l_Bints = self.propag_MPC()
            dict_waists = self.waists()
            self.update_entry('Waist_lens', dict_waists['Waist on lens'])

        try:
            for art in self.l_arts:
                art.remove()
        except AttributeError:
            self.l_arts = []
        self.l_arts = []
        for i in range(self.N):
            # print(f'Rebond {i}, waist {l_waists[i]*1e6}µm')
            x, y = pol2cart(l_Rs[i], l_thetas[i])
            x *= 1e3
            y *= 1e3
            temp = self.axes[i%2].add_artist(plt.Circle([x, y], l_waists[i]*3e3, color='k'))
            # self.circles_list[i].update({'center':[x,y], 'radius':l_waists[i]*3e3})
            self.l_arts.append(temp)
            
            temp = self.axes[i%2].annotate(str(i), xy= (x, y), c='r')
            self.l_arts.append(temp)
        self.fig.canvas.draw()
        # self.fig.canvas.flush_events()




        # try:
        # print(dict_waists)
        for i, name in enumerate(self.outputs):
            # print(i, name)
            # self.outputs[name].set(dict_waists[name])
            self.outputs[name]['text'] = dict_waists[name]
        # except AttributeError:
        #     print('Attribute Error :(')
        #     pass


        # print(self.outputs_bis)
        # print(f'{self.Bint/self.N/np.pi:.2f}')
        self.outputs_bis['Integrale B par rebond']['text'] = f'{self.Bint/self.N/np.pi:.2f}'
        self.outputs_bis['Integrale B totale']['text'] = f'{self.Bint/np.pi:.2f}'
        self.outputs_bis['Ratio de compression']['text'] = f'{self.Bint/np.pi:.2f}'
        self.outputs_bis['tau_in']['text'] = f'{self.tau0*1e15:.2f}'
        self.outputs_bis['tau_out_B']['text'] = f'{self.tau0*1e15/(self.Bint/np.pi):.2f}'
        self.outputs_bis['Integrale B Viotti par rebond']['text'] = f'{self.BintViotti/np.pi/self.N:.2f}'
        self.outputs_bis['Integrale BViotti totale']['text'] = f'{self.BintViotti/np.pi:.2f}'
        self.outputs_bis['Ratio de compression Viotti']['text'] = f'{self.BintViotti/np.pi:.2f}'
        self.outputs_bis['tau_out_B Viotti']['text'] = f'{self.tau0*1e15/(self.BintViotti/np.pi):.2f}'

class LimitedFloatEntry(ttk.Entry):
    '''A new type of Entry widget that allows you to set limits on the entry'''
    def __init__(self, master=None, default=0, bounds=[-1,1], **kwargs):
        self.var = tk.StringVar(master, default)
        self.var.trace('w', self.validate)
        self.get = self.var.get
        self.bounds  = bounds
        self.old_value = 0
        ttk.Entry.__init__(self, master, textvariable=self.var, **kwargs)

    def validate(self, *args):
        # print(self.bounds)
        try:
            value = self.get()
            print(value)
            # special case allows for an empty entry box
            if value not in ('', '-') and not self.bounds[0] <= float(value) <= self.bounds[1]:
                raise ValueError
            self.old_value = value
        except ValueError:
            print('c')
            self.set(self.old_value)

    def set(self, value):
        self.delete(0, tk.END)
        self.insert(0, str(value))
    
    def value(self):
        return(float(self.get()))




##########Code à proprement parler##########

reflectivity = 0.99
n0 = 1
n2 = 0.97e-23
n2_outside = 3e-24
wl = 1030
Entry_widgets = {}
MPC_obj = MPC_tk(entries=Entry_widgets, reflectivity=reflectivity, n0=n0, n2=n2, n2_outside=n2_outside, wl=wl)
window = tk.Tk()

style = ttk.Style()
# style.configure("TLabel")
style.configure('.', font=('Arial', 16))

frame_entries = tk.Frame(master=window, relief='ridge', borderwidth=5, width=200, height=100)
frame_MPC = tk.Frame(master=window, relief='ridge', borderwidth=5, width=100, height=50)
ttk.Label(master=frame_entries, text='Entries').grid(row=0, column=0)




##### Entries
Label_titles = {
    'Energy':'Energy (mJ)', 'Duration':'tau_in (fs)', 'RoC':'RoC (m)', 'Radius':'Radius (mm)', 'N':'N', 'L':'L (m)',
    'Waist_foc':'Waist at focus (µm)', 'Waist_lens':'Waist at lens (µm)', 
    'Pressure':'Pressure (bar)', 'x':'x (mm)', 'y':'y (mm)', 'thetax':'thetax (mrad)', 'thetay':'thetay (mrad)', 
    'M2':'M2', 'd_window':'d center - window (m)', 'd_lens':'d window - lens (m)', 'f':'f lens (m)'
    #'Optimize angle', 'Optimal waist', 'NL_propag', 'Propag from outside'
}

Entry_values = {
    'Energy':4, 'Duration':450, 'RoC':0.75, 'Radius':25.4, 'N':15, 'L':1.481,
    'Waist_foc':165, 'Waist_lens':1000, 
    'Pressure':1, 'x':20, 'y':0, 'thetax':0, 'thetay':0, 
    'M2':1, 'd_window':0.5, 'd_lens':2.5, 'f':3
}

Entry_bounds = {
    'Energy':[0, 10], 'Duration':[0, 1000], 'RoC':[0.1,2], 'Radius':[0, 100], 'N':[1, 30], 'L':[0, 1.49],
    'Waist_foc':[0, 2000], 'Waist_lens':[0, 50000], 
    'Pressure':[0, 10], 'x':[-100,100], 'y':[-100,100], 'thetax':[-50, 50], 'thetay':[-50,50], 
    'M2':[0, 1.2], 'd_window':[0, 10], 'd_lens':[0, 10], 'f':[0, 10]
}



##### Entry buttons
CBtn_list = ['NL_propag', 'Propag from outside']
Btn_list = ['Optimize angle', 'Optimal waist']
Btn_list_cmd = [MPC_obj.optimize_injection, MPC_obj.optimize_waist]
MPC_obj.cbtn_vars = {'NL_propag':tk.BooleanVar(), 'Propag from outside':tk.BooleanVar()}

# MPC_tk.cbtn_vars['NL_propag'].set(False)  #Par défaut c'est false
# MPC_tk.cbtn_vars['Propag from outside'].set(False) 

current_row = 1
for i, label in enumerate(Label_titles):
    ttk.Label(master=frame_entries, text=Label_titles[label]).grid(row=current_row, column=0, sticky='w')
    Entry_widgets[label] = LimitedFloatEntry(master=frame_entries, bounds=Entry_bounds[label], default=Entry_values[label], font=('Arial', 16))
    
    Entry_widgets[label].grid(row=i+1, column=1, sticky='e')
    current_row += 1

for i, btn_name in enumerate(Btn_list):
    ttk.Button(master=frame_entries, text=btn_name, command=Btn_list_cmd[i]).grid(row=current_row, column=0, sticky='w')
    current_row += 1

for i, cbtn_name in enumerate(CBtn_list):
    cbtn = ttk.Checkbutton(master=frame_entries, variable=MPC_obj.cbtn_vars[cbtn_name], text=cbtn_name)
    cbtn.grid(row=current_row, column=0, sticky='w')

    current_row += 1



##### Axes
ttk.Label(master=frame_MPC, text='MPC').pack()
fig, axes = plt.subplots(1,2, figsize=(10,4), sharex=True, sharey=True)
axes[0].set_ylabel('y (mm)')
axes[0].set_xlabel('x (mm)')
axes[1].set_xlabel('x (mm)')
figure_canvas = FigureCanvasTkAgg(fig, frame_MPC)
figure_canvas._tkcanvas.pack(fill=tk.BOTH, expand=1)

# print(Entry_widgets)


##### Lower grid (text output)
frame_output_labels = tk.Frame(master=window, relief='ridge', borderwidth=5)
ttk.Label(master=frame_output_labels, text='OUTPUTS').grid(row=0, column=0, sticky='w')


list_output_names = ['Sigma', 'Waist on mirrors', 'Waist at focus', 'Waist on window', 'Waist on lens', 'Waist for alignment', 'Fluence on mirrors']
list_output_units = {'Sigma':None, 'Waist on mirrors':'µm', 'Waist at focus':'µm', 'Waist on window':'µm', 'Waist on lens':'µm', 'Waist for alignment':'µm', 'Fluence on mirrors':'J/cm^2'}
list_output_values = {}


for i, name in enumerate(list_output_names):
    ttk.Label(master=frame_output_labels, text=name).grid(row=i+1, column=0, sticky='w')
    list_output_values[name] = ttk.Label(master=frame_output_labels, text=0)
    list_output_values[name].grid(row=i+1, column=1, sticky='w')
    # ttk.Label(master=frame_output_labels, text=0).grid(row=i+1, column=1, sticky='w')
    ttk.Label(master=frame_output_labels, text=list_output_units[name]).grid(row=i+1, column=3, sticky='w')
MPC_obj.outputs = list_output_values



frame_output_bis = tk.Frame(master=window, relief='ridge', borderwidth=5)

# ttk.Label(master=frame_output_labels, text='OUTPUTS').grid(row=0, column=0, sticky='w')


list_output_names_bis = ['Integrale B par rebond', 'Integrale B totale', 'Ratio de compression', 'tau_in', 'tau_out_B', 
                     'Integrale B Viotti par rebond', 'Integrale BViotti totale', 'Ratio de compression Viotti', 'tau_out_B Viotti']#, 'Elargissement spectral', 'tau_out_b']
list_output_units_bis = {'Integrale B par rebond':'pi', 'Integrale B totale':'pi', 'Ratio de compression':None, 'tau_in':'fs', 'tau_out_B':'fs', 
                     'Integrale B Viotti par rebond':'pi', 'Integrale BViotti totale':'pi', 'Ratio de compression Viotti':None, 'tau_out_B Viotti':'fs'}#, 'Elargissement spectral', 'tau_out_b']

list_output_values_bis = {}


for i, name in enumerate(list_output_names_bis):
    ttk.Label(master=frame_output_bis, text=name).grid(row=i+1, column=0, sticky='w')
    list_output_values_bis[name] = ttk.Label(master=frame_output_bis, text=0)
    list_output_values_bis[name].grid(row=i+1, column=1, stick='w')
#     list_output_values[name].grid(row=i+1, column=1, sticky='w')
#     # ttk.Label(master=frame_output_labels, text=0).grid(row=i+1, column=1, sticky='w')
    ttk.Label(master=frame_output_bis, text=list_output_units_bis[name]).grid(row=i+1, column=3, sticky='w')
MPC_obj.outputs_bis = list_output_values_bis


##### Lower buttons
btn_update = ttk.Button(window, text='Update', command=MPC_obj.update_entries).grid(row=3, column=0, sticky='w')
btn_quit = ttk.Button(window, text="Quit", command=window.quit).grid(row=3, column=1, sticky='w')




MPC_obj.update_entries()
MPC_obj.init_axes(fig, axes)

frame_entries.grid(row=0, column=0, sticky='nw')
frame_MPC.grid(row=0, column=1, sticky='nw')
frame_output_labels.grid(row=1, column=0, sticky='nw')
frame_output_bis.grid(row=1, column=1, sticky='nw')
window.rowconfigure((0,1), weight=1, minsize=50)
window.columnconfigure((0,1), weight=1, minsize=50)
window.protocol("WM_DELETE_WINDOW", window.quit)

window.geometry()

window.mainloop()
window.destroy()
