import numpy as np
import matplotlib.pyplot as plt

'''
Rechenaufgabe:
'''
mu0=4*np.pi*1e-7
c=3e8

E=820e9 #eV
p=E/c
L=9 #m
theta=15e-3 #rad
Bahnradius=L/theta # =600m
B=E/Bahnradius/c # =4.56T
        #angenommen, es gibt nur einenen Draht pro Seite:
I=B/2 * 2*np.pi*0.04 /mu0
F=L*mu0*I**2/(2*np.pi*2*0.04)

print(f"Stromstärke für einen Draht pro Seite: {I/1e3:.1f}kA")
print(f"Abstoßende Kraft: {F/1e6:.1f}MN")

'''
Programmieraufgabe
'''

# Parameter der Drähte
N=31        #Drähte pro Seite
R=0.04      #Abstand zur Achse
phi=0.25 * np.pi    #Winkel, die jede Stromschale einnimmt
j=I/phi    #Stromdichte

delta_phi=phi/(N-1)


#Orte der Drähte festlegen
x_Draht=np.zeros(2*N)                       # Speicher allokieren
y_Draht=np.zeros(2*N)
I_Draht=np.zeros(2*N)
Draht_winkel=np.linspace(-phi/2,phi/2,N)    # äquidistante Winkel festlegen

x_Draht[:N]=R*np.cos(Draht_winkel)         # Orte ausrechnen (Polarkoordinaten)
y_Draht[:N]=R*np.sin(Draht_winkel)
x_Draht[N:]=-x_Draht[:N]                  # andere Seite ausrechnen (ich habe die Symmetrie ausgenutzt)
y_Draht[N:]=y_Draht[:N]
I_Draht[:N]=j*delta_phi
I_Draht[N:]=-j*delta_phi

plt.figure(1)
plt.clf()
plt.plot(x_Draht,y_Draht,'X')
plt.axis('equal')

#Raster für die Orte der Auswertungspunkte sowie deren nachher berechnetes Magnetfeld anlegen
n=31        #Anzahl Punkte pro Achse (ungerade, damit (0,0) dabei ist), also n^2 insgesamt
l=0.08      #maximale Ablage zum Zentrum, in dem das B-Feld ausgewertet werden soll.
X,Y=np.meshgrid(np.linspace(-l,l,n),np.linspace(-l,l,n))

B_x=np.zeros((n,n))
B_y=np.zeros((n,n))
B_z=np.zeros((n,n))
plt.plot(X,Y,'ok',ms=0.2)
maxB_z=0
for i in range(n):
    for j in range(n):
        for k in range(2*N):
            d_vec=[X[i,j]-x_Draht[k],Y[i,j]-y_Draht[k],0]
            d_length=np.sqrt(d_vec[0]**2+d_vec[1]**2)
            B_Beitrag=-mu0/(2*np.pi*d_length**2)*np.cross([0,0,I_Draht[k]],d_vec)
            B_x[i,j]+=B_Beitrag[0]
            B_y[i,j]=B_y[i,j]+B_Beitrag[1]
plt.figure(1)
plt.clf
plt.quiver(X,Y,B_x,B_y)    
plt.axis('equal')
plt.figure(2)      
plt.plot(B_y[int((n-1)/2),:],label='$\Phi_0=$'+str(int(phi/np.pi*180))+'°') 
plt.legend()
plt.savefig('field.pdf')