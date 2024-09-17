import os
import matplotlib.pyplot as plt
import ngsolve as ng
import numpy as np
from netgen.geom2d import SplineGeometry
from ngsolve import ngsglobals
from scipy import sparse
ngsglobals.msg_level = 0


def offset_cyl_ROM(method,num_vecs,h,dt,dt0,order):
    completed = True

    nu = 0.01
    eps_val=1e-8
    vids=False
    arrs=True

    print_rate=int(.01/dt)
    if print_rate==0:
        print_rate=1
    

    num_snaps=int(.67/dt) # two periods about

    t_init = 12
    t_final = 16.0
    t = t_init-dt
    num_t_steps = int((t_final - t_init) / dt)+1

    info_string='_o_'+str(order)+'_'+str(num_vecs)+'_dt_'+str(dt)
    if arrs is True and not os.path.exists("./offset_arrs/"+info_string):
        os.makedirs("offset_arrs/"+info_string)
    if vids is True and not os.path.exists("./offset_vid/"+info_string):
        os.makedirs("offset_vid/"+info_string)

    #Set up Geometry
    geo = SplineGeometry()
    geo.AddCircle(
        (0, 0), r=1, bc="wall"
    )
    geo.AddCircle(
        (0.5, 0.0), r=0.1, leftdomain=0, rightdomain=1, bc="cyl", maxh=h / 5
    )
    mesh = ng.Mesh(geo.GenerateMesh(maxh=h))
    mesh.Curve(3)

    #TH Element Space
    V = ng.VectorH1(mesh, order=2, dirichlet="wall|cyl")
    Q = ng.H1(mesh, order=1)
    X = ng.FESpace([V, Q])
    X2= ng.VectorH1(mesh, order=2, dirichlet="wall|cyl")

    print("The total number of velocity degrees of freedom: " + str(V.ndof))
    print("The total number of pressure degrees of freedom: " + str(Q.ndof))
    print("The total number of degrees of freedom: " + str(Q.ndof + V.ndof))

    print(info_string)
    # Define test and trial functions
    u, p = X.TrialFunction()
    v, q = X.TestFunction()

    # Gridfunctions
    up_n0 = ng.GridFunction(X) #un
    up_n1 = ng.GridFunction(X) #unp1
    up_n2 = ng.GridFunction(X) #unp2
    up_ab2 = ng.GridFunction(X) #2unp1-un
    up_XL = ng.GridFunction(X) #xtra lagged

    up_diff = ng.GridFunction(X)

    #ROM Grid Functions
    up_ROM=[ng.GridFunction(X) for i in range(0,num_vecs)]
    #Average velocity
    up_avg = ng.GridFunction(X) #xtra lagged

    # Functions for lift and drag computation
    drag_test = ng.GridFunction(X)
    drag_test.components[0].Set(
        ng.CoefficientFunction((0, -1.0)), definedon=mesh.Boundaries("cyl")
    )
    lift_test = ng.GridFunction(X)
    lift_test.components[0].Set(
        ng.CoefficientFunction((-1.0, 0.0)), definedon=mesh.Boundaries("cyl")
    )

    # Velocity solution at previous iteration
    u0 = ng.CoefficientFunction(up_n1.components[0])
    u1 = ng.CoefficientFunction(up_n1.components[0])
    u2 = ng.CoefficientFunction(up_n2.components[0])
    p0 = ng.CoefficientFunction(up_n0.components[1])
    u_ab2 = ng.CoefficientFunction(up_ab2.components[0])  # 2u1 - u2

    #Xtra Lagged Option
    Us=ng.CoefficientFunction(up_XL.components[0])

    # Define time array for plotting purposes
    time_val_arr = np.array([])


    # Define parameters
    k = ng.Parameter(1.0)
    if t<=1:
        k.Set(t)
        
    # BC?
    uin = ng.CoefficientFunction(
        (0, 0)
    )

    #Forcing function
    frhs = ng.CoefficientFunction(
        (-4*k*ng.y*(1-ng.x*ng.x-ng.y*ng.y) ,  4*k*ng.x*(1-ng.x*ng.x-ng.y*ng.y))
    )

    #Load ICs
    up_n2.vec.FV().NumPy()[:] = np.loadtxt('save_vecs/h_'+str(h)+'_dt_'+str(dt0)+'_uMinus1.txt')


    # Lists for holding the drag and lift at each timestep
    drag_arr_hold = []
    lift_arr_hold = []
    udiff_arr_hold = []
    gradu_arr_hold = []
    NL_arr_hold = []
    ND_arr_hold = []


    # Set up weak form of the NSE
    stokes = (
        nu * ng.InnerProduct(ng.grad(u), ng.grad(v))
        + ng.div(u) * q
        - ng.div(v) * p
        + eps_val * p * q
    )

    # BE
    udt = ng.InnerProduct(u, v)

    #Xtra lagged Implicit Term
    conv_XL = ng.InnerProduct(ng.grad(u) * Us, v)
    if order ==1:
        up_XL.vec.FV().NumPy()[:] = up_n1.vec.FV().NumPy()[:] 
        # Set up left hand side of BDF2-AB2
        NSE = 1.0 / (1.0 * dt) * udt + stokes + conv_XL
        a = ng.BilinearForm(X)
        a += ng.SymbolicBFI(NSE)
        a.Assemble()

        # Set up righthand side of NSE system
        f = ng.LinearForm(X)
        f += (1.0 / dt * u1 * v +frhs*v +  ng.grad(up_n1.components[0]) * (Us-u1)*v ) * ng.dx
        f.Assemble()

    if order == 2:
        up_XL.vec.FV().NumPy()[:] = (2 * up_n1.vec.FV().NumPy()[:] - up_n2.vec.FV().NumPy()[:])
        # Set up left hand side of BDF2-AB2
        NSE = 3.0 / (2.0 * dt) * udt + stokes + conv_XL
        a = ng.BilinearForm(X)
        a += ng.SymbolicBFI(NSE)
        a.Assemble()

        # Set up righthand side of NSE system
        f = ng.LinearForm(X)
        f += (2.0 / dt * u1 - (1.0 / (2.0 * dt)) * u2 + frhs) * v * ng.dx
        f += ng.InnerProduct(ng.grad(up_ab2.components[0])*(Us-u_ab2), v) * ng.dx
        f.Assemble()


    if dt0 == dt:
        up_n1.vec.FV().NumPy()[:] = np.loadtxt('save_vecs/h_'+str(h)+'_dt_'+str(dt0)+'_u0.txt')
        up_ab2.vec.FV().NumPy()[:] = (2 * up_n1.vec.FV().NumPy()[:] - up_n2.vec.FV().NumPy()[:])
    else:
        # Set up left hand side of imex euler
        conv_euler = ng.InnerProduct(ng.grad(u) * u2, v)
        NSE_ic = 1.0 / dt * udt + (stokes + conv_euler)
        a_ic = ng.BilinearForm(X)
        a_ic += ng.SymbolicBFI(NSE_ic)
        a_ic.Assemble()

        # Set up right hand side of imex euler
        f_ic = ng.LinearForm(X)
        f_ic += (1.0 / dt * u2 * v + frhs*v) * ng.dx
        f_ic.Assemble()
        ng.solvers.BVP(bf=a_ic, lf=f_ic, gf=up_n1, pre=None, print=False, inverse="umfpack")

        up_ab2.vec.FV().NumPy()[:] = (2 * up_n1.vec.FV().NumPy()[:] - up_n2.vec.FV().NumPy()[:])
        up_XL.vec.FV().NumPy()[:] = (2 * up_n1.vec.FV().NumPy()[:] - up_n2.vec.FV().NumPy()[:])


    if vids:
        vtk = ng.VTKOutput(mesh,coefs=[up_n0.components[0],up_n0.components[1]],names=["vel","press"],filename="offset_vid"+info_string+"/video",subdivision=2)
        vtk.Do()
        

    snapshots = np.zeros((num_snaps,V.ndof))
    count = 0
    udiff=0
    # Begin Time loop

    for jj in range(num_t_steps - 1): 
        if jj < num_snaps:
            if order == 1:
                snapshots[jj,:]=up_n0.components[0].vec.FV().NumPy()[:]
            if order == 2:
                snapshots[jj,:]=up_ab2.components[0].vec.FV().NumPy()[:]

        if jj==num_snaps and method ==2:
            e_vals,e_vecs= get_pod(num_snaps,num_vecs,snapshots,X2,V.ndof)
            for i in range(0,num_vecs):
                up_ROM[i].components[0].vec.FV().NumPy()[:]=e_vecs[:,i]

        if t<=1:
            k.Set(t)
        
        # Solve NSE system
        a.Assemble()
        f.Assemble()

        ng.solvers.BVP(bf=a, lf=f, gf=up_n0, pre=None, print=False, inverse="umfpack")
        # Update ab2 term
        up_ab2.vec.FV().NumPy()[:] = (
            2 * up_n0.vec.FV().NumPy()[:] - up_n1.vec.FV().NumPy()[:]
        )

            

        if order ==2:
            if jj<num_snaps:
                up_XL.vec.data=up_ab2.vec.data

            else:
                up_XL.components[0].vec.FV().NumPy()[:]=0.0
                for i in range(0,num_vecs):
                    up_XL.components[0].vec.FV().NumPy()[:]+=ng.Integrate (ng.InnerProduct(ng.grad(up_ROM[i].components[0]) ,ng.grad(up_ab2.components[0])), mesh)*up_ROM[i].components[0].vec.FV().NumPy()[:]

            up_diff.vec.FV().NumPy()[:]=up_ab2.vec.FV().NumPy()[:]-up_XL.vec.FV().NumPy()[:]

        if order ==1:
            if jj<num_snaps:
                up_XL.vec.data=up_n0.vec.data
            else:
                up_XL.components[0].vec.FV().NumPy()[:]=0.0
                for i in range(0,num_vecs):
                    up_XL.components[0].vec.FV().NumPy()[:]+=ng.Integrate (ng.InnerProduct(ng.grad(up_ROM[i].components[0]) ,ng.grad(up_n0.components[0])), mesh)*up_ROM[i].components[0].vec.FV().NumPy()[:]

            up_diff.vec.FV().NumPy()[:]=up_n0.vec.FV().NumPy()[:]-up_XL.vec.FV().NumPy()[:]




        if jj %print_rate==0:
            if order ==1:
                drag_val= calculate_ld_be(up_n0,up_n1,drag_test,nu,dt,mesh)
                lift_val= calculate_ld_be(up_n0,up_n1,lift_test,nu,dt,mesh)
                up_diff.vec.FV().NumPy()[:]=up_n0.vec.FV().NumPy()[:]-up_XL.vec.FV().NumPy()[:]
            if order ==2:
                drag_val= calculate_ld_bdf2(up_n0,up_n1,up_n2,drag_test,nu,dt,mesh)
                lift_val= calculate_ld_bdf2(up_n0,up_n1,up_n2,lift_test,nu,dt,mesh)
                up_diff.vec.FV().NumPy()[:]=up_ab2.vec.FV().NumPy()[:]-up_XL.vec.FV().NumPy()[:]

            
            
            udiff_func=up_diff.components[0]
            udiff=np.sqrt(ng.Integrate (ng.InnerProduct( ng.grad(udiff_func),(ng.grad(udiff_func))), mesh))
            gradu=ng.Integrate (ng.InnerProduct( ng.grad(up_n0.components[0]),(ng.grad(up_n0.components[0]))), mesh)
            if order ==1:
                ND=ng.Integrate (ng.InnerProduct( u0-u1,u0-u1), mesh)/(2.0*dt)
                NL=ng.Integrate(ng.InnerProduct(ng.grad(up_n0.components[0])*(Us-u1), up_n0.components[0]),mesh)

                
            if order ==2:
                ND=ng.Integrate (ng.InnerProduct( u0-2*u1+u2,u0-2*u1+u2), mesh)/(4.0*dt)
                NL=ng.Integrate(ng.InnerProduct(ng.grad(up_ab2.components[0])*(Us-u_ab2), up_n0.components[0]),mesh)


            time_val_arr = np.append(time_val_arr, t)
            drag_arr_hold.append(drag_val)
            lift_arr_hold.append(lift_val)
            udiff_arr_hold.append(udiff)
            gradu_arr_hold.append(gradu)
            NL_arr_hold.append(NL)
            ND_arr_hold.append(ND)
            
            if abs(int(t)-t )< dt/10:
                print(f"The current time is: {t}")
            # print("drag value: " + str(drag_val))
        if order ==2:
            up_n2.vec.data = up_n1.vec.data  
        up_n1.vec.data = up_n0.vec.data 
           
            

        if abs(drag_val) > 1e6:
            print("drag value: " + str(drag_val))

            np.savetxt('offset_arrs/'+info_string+'/time'+info_string+'.txt',time_val_arr)
            np.savetxt('offset_arrs/'+info_string+'/drag'+info_string+'.txt',drag_arr_hold)
            np.savetxt('offset_arrs/'+info_string+'/lift'+info_string+'.txt',lift_arr_hold)
            np.savetxt('offset_arrs/'+info_string+'/udff'+info_string+'.txt',udiff_arr_hold)
            np.savetxt('offset_arrs/'+info_string+'/gradu'+info_string+'.txt',gradu_arr_hold)
            np.savetxt('offset_arrs/'+info_string+'/NL'+info_string+'.txt',NL_arr_hold)
            np.savetxt('offset_arrs/'+info_string+'/ND'+info_string+'.txt',ND_arr_hold)
            completed = False
            break

        
    if order ==1:
        drag_val= calculate_ld_be(up_n0,up_n1,drag_test,nu,dt,mesh)
        lift_val= calculate_ld_be(up_n0,up_n1,lift_test,nu,dt,mesh)
        up_diff.vec.FV().NumPy()[:]=up_n0.vec.FV().NumPy()[:]-up_XL.vec.FV().NumPy()[:]
    if order ==2:
        drag_val= calculate_ld_bdf2(up_n0,up_n1,up_n2,drag_test,nu,dt,mesh)
        lift_val= calculate_ld_bdf2(up_n0,up_n1,up_n2,lift_test,nu,dt,mesh)
        up_diff.vec.FV().NumPy()[:]=up_ab2.vec.FV().NumPy()[:]-up_XL.vec.FV().NumPy()[:]
    udiff=np.sqrt(ng.Integrate (ng.InnerProduct( ng.grad(udiff_func),(ng.grad(udiff_func))), mesh))
    
    time_val_arr = np.append(time_val_arr, t)
    drag_arr_hold.append(drag_val)
    lift_arr_hold.append(lift_val)
    udiff_arr_hold.append(udiff)
    np.savetxt('offset_arrs/'+info_string+'/time'+info_string+'.txt',time_val_arr)
    np.savetxt('offset_arrs/'+info_string+'/drag'+info_string+'.txt',drag_arr_hold)
    np.savetxt('offset_arrs/'+info_string+'/lift'+info_string+'.txt',lift_arr_hold)
    np.savetxt('offset_arrs/'+info_string+'/udff'+info_string+'.txt',udiff_arr_hold)
    np.savetxt('offset_arrs/'+info_string+'/gradu'+info_string+'.txt',gradu_arr_hold)
    np.savetxt('offset_arrs/'+info_string+'/NL'+info_string+'.txt',NL_arr_hold)
    np.savetxt('offset_arrs/'+info_string+'/ND'+info_string+'.txt',ND_arr_hold)

    return completed



def calculate_ld_be(up0,up1,drag_test,nu,dt,mesh):
    drag_val = (
        1.0/ dt
        * ng.Integrate(
            ng.InnerProduct(
                up0.components[0] - up1.components[0], drag_test.components[0]
            ),
            mesh,
        )
        - ng.Integrate(up0.components[1] * ng.div(drag_test.components[0]), mesh)
        + nu
        * ng.Integrate(
            ng.InnerProduct(
                ng.grad(up0.components[0]), ng.grad(drag_test.components[0])
            ),
            mesh,
        )
        + ng.Integrate(
            ng.InnerProduct(
                ng.grad(up0.components[0]) * up0.components[0],
                drag_test.components[0],
            ),
            mesh,
        )
    )
    return drag_val

def calculate_ld_bdf2(up_n0,up_n1,up_n2,drag_test,nu,dt,mesh):
    drag_val = (
            1.0
            / (2.0 * dt)
            * ng.Integrate(
                ng.InnerProduct(
                    3 * up_n0.components[0]
                    - 4 * up_n1.components[0]
                    + up_n2.components[0],
                    drag_test.components[0],
                ),
                mesh,
            )
            - ng.Integrate(up_n0.components[1] * ng.div(drag_test.components[0]), mesh)
            + nu
            * ng.Integrate(
                ng.InnerProduct(
                    ng.grad(up_n0.components[0]), ng.grad(drag_test.components[0])
                ),
                mesh,
            )
            + ng.Integrate(
                ng.InnerProduct(
                    ng.grad(up_n0.components[0]) * up_n0.components[0],
                    drag_test.components[0],
                ),
                mesh,
            )
        )
    return drag_val

def get_pod(nsnsh,nbasis,H_v,X,vdof):

	# Pod construction
	print("Begin POD construction")
	u2= X.TrialFunction()
	v2= X.TestFunction()
	
	H_v = 1.0/np.sqrt(nsnsh) * np.transpose(H_v)
	Ht_v = np.transpose(H_v)
	
	NSE = ng.InnerProduct(ng.grad(u2),ng.grad(v2))
	a = ng.BilinearForm(X)
	
	a += ng.SymbolicBFI(NSE)
	a.Assemble()


	# row_A,col_A,val_A = as_backend_type(A_1).data()
	rows,cols,vals = a.mat.COO()
	sA = sparse.csr_matrix((vals,(rows,cols)))

	print(np.shape(sA))
	print(np.shape(H_v))

	C_temp_v = sA.dot(H_v)
	C_v = Ht_v.dot(C_temp_v)


	#Velocity Pod
	j = nbasis #max pod vecs
	vals_v, vecs_v = sparse.linalg.eigsh(C_v, k=j)
	eigvals_v = np.zeros((j,1))
	eigvecs_v = np.zeros((vdof,j))
	for i in range(0,j):
		#print nRom
		eigvals_v[i] = vals_v[j-i-1]
		temp_v =  vecs_v[:,j-i-1]
		fin_v = H_v.dot(temp_v)
		fin_v = fin_v/np.sqrt(eigvals_v[i])
		eigvecs_v[:,i] = fin_v[:]
	print("The eigenvalues associated with our velocity POD matrix are")
	print(eigvals_v)

	# filename_v = './POD_vecs/velocityMatBE.txt'
	# filename2_v = './POD_vecs/POD_vals_velocityBE.txt'
	# np.savetxt(filename_v,eigvecs_v);
	# np.savetxt(filename2_v,eigvals_v);
	return eigvals_v, eigvecs_v
