import numpy as np
import matplotlib.pyplot as plt

def initialize(particlePerCell, nodeNum, domainHeight, dt, simulationTime, randomParticle):
    if randomParticle:
        particleNum = (nodeNum - 1) * (particlePerCell)   # Number of particles in 'x' dimension # 1D
        dx = domainHeight / (nodeNum - 1)  # Node spacing
        dp = domainHeight / (particleNum - 1)  # Particle spacing
        xI = np.arange(0, domainHeight + dx - 1e-15, dx)  # Node locations
        np.random.seed(0)  # Set random seed for reproducibility
        xp = np.zeros(particleNum)
        for i in range(nodeNum - 1):
            for ppc in range (particlePerCell):
                if i == 0: # begin
                    xp[particlePerCell*i + ppc] = dx * np.random.rand(1) + dx * i
                    xp[particlePerCell*i] = 0
                elif i == nodeNum - 2: # end
                    xp[particlePerCell*i + ppc] = dx * np.random.rand(1) + dx * i
                    xp[particlePerCell*i + particlePerCell - 1] = domainHeight
                else: # middle
                    xp[particlePerCell*i + ppc] = dx * np.random.rand(1) + dx * i
    else:
        particleNum = (nodeNum - 1) * (particlePerCell)  # Number of particles in 'y' dimension # 1D
        # particleNum = (nodeNum - 1) * (particlePerCell - 1) + 1 # paticle will overlap the node
        dx = domainHeight / (nodeNum - 1)  # Node spacing
        dp = domainHeight / (particleNum - 1)  # Particle spacing
        xI = np.arange(0, domainHeight + dx - 1e-12, dx)  # Node locations
        xp = np.zeros(particleNum)
        for i in range(nodeNum - 1): # make particle between the node
            for ppc in range (particlePerCell):
                if i == 0:
                    xp[particlePerCell*i + ppc] = dx * ppc / (particlePerCell) + dx * i
                    xp[particlePerCell*i] = 0
                elif i == nodeNum - 2:
                    xp[particlePerCell*i + ppc] = dx * (ppc+1) / particlePerCell + dx * i
                    xp[particlePerCell*i + particlePerCell - 1] = domainHeight
                else:
                    xp[particlePerCell*i + ppc] = dx * (ppc+1) / (particlePerCell+1) + dx * i
        # xp = np.linspace(0, domainHeight, particleNum)  # Particle locations

    # Node displacement
    uI = np.zeros_like(xI)
    # Node pressure
    pI = np.zeros_like(xI)
    # Total number of timesteps
    timeStepNum = simulationTime / dt
    # Midpoint between each particle
    midpoint = (xp[1:] - xp[:-1]) / 2 + xp[:-1]
    # Calculate each particle volume
    vp = midpoint[1:] - midpoint[:-1]
    vp = np.concatenate(([midpoint[0]], vp, [domainHeight - midpoint[-1]]))
    # Initialize normal vector
    n = np.zeros(particleNum)
    n[0] = -1  # Normal vector at y = 0 (Dirichlet BC)
    n[-1] = 1  # Normal vector at y = Ly (traction BC)

    return particleNum, dx, dp, xp, vp, xI, uI, pI, n, timeStepNum

def initialize_terzaghi(E, nu, kh, gammaf, nf, Kf):
    K = E / (3 * (1 - 2 * nu))  # Bulk modulus
    G = E / (2 * (1 + nu))  # Shear modulus
    kf = kh / gammaf  # Permeability
    Ks = K * 1e10  # Solid grain bulk modulus
    alpha = 1 - K / Ks  # Biot's coefficient
    Q_inv = (alpha - nf) / Ks + nf / Kf  # 1 / Q
    return kf, Ks, alpha, Q_inv, K, G

def GetRK(xI, dx, xp, supportNorm): # Node location, Node spacing, Particle location, Normalized RK approximation support size
    nodeNum = len(xI)                   # Number of background grid nodes
    particleNum = len(xp)               # Number of locations to evaluate the RK shape functions at
    w = np.zeros((nodeNum, particleNum))  # Initialize vector for kernel function weights
    phi = np.zeros((nodeNum, particleNum))  # Initialize vector for RK shape functions
    dphi = np.zeros((nodeNum, particleNum))  # Initialize vector for RK shape function gradient
    support = supportNorm * dx           # Define support size for this discretization
    support0 = support                   # "Initial" support size, only necessary if it will be changed

    for idx in range(particleNum):
        M = np.zeros((2, 2))                     # Initialize moment matrix
        for I in range(nodeNum):
            z = abs(xp[idx] - xI[I]) / support  # Normalized distance between the particle to each node

            if 0 <= z and z <= 0.5:  # Define cubic b-spline kernel function weights
                w[I, idx] = 2/3 - 4*z**2 + 4*z**3
            elif 0.5 <= z and z < 1:
                w[I, idx] = 4/3 - 4*z + 4*z**2 - (4/3)*z**3
            else:
                w[I, idx] = 0

            PxI = np.array([1, xI[I] - xp[idx]])  # Define P(xI - xp)
            if w[I, idx] != 0:  # If the kernel function weight is not zero
                M = M + w[I, idx] * np.outer(PxI, PxI)  # Define the moment matrix
        for J in range(nodeNum):  # Define RK approximation
            PxI = np.array([1, xI[J] - xp[idx]])
            Ploc = np.array([1, 0])
            PlocGrad = np.array([0, 1])
            phi[J, idx] = w[J, idx] * np.dot(Ploc, np.linalg.solve(M, PxI))
            dphi[J, idx] = w[J, idx] * np.dot(PlocGrad, np.linalg.solve(M, PxI))

    return phi, dphi

def get_vc(phiU, dphiU, nodeNum, vp, n):
    eps = 1e-15
    ksiNum = np.zeros(nodeNum)
    ksiDen = np.zeros(nodeNum)

    for I in range(nodeNum):
        ksiNum[I] = np.dot(dphiU[I, :], vp)
        ksiNum[I] -= np.dot(phiU[I, :], n) * 1
        localParticle = np.abs(phiU[I, :]) > eps
        ksiDen[I] = np.sum(vp[localParticle])

    ksi = -ksiNum / ksiDen
    return ksi, eps

def pressure_stabilize(dx, dp, Ks, Kf, kf, alpha, nf, K, G, dt, p, xp, vp, xI, nodeNum, particleNum, supportNorm,particlePerCell):
    if p == 0 or p == particleNum - 1: # front & end
        h1 = dx / 2 # grid node
        h2 = dp / 2 # material particle
    else:
        h1 = dx
        h2 = dp
    Mb = Ks * Kf / (Kf * (alpha - nf) + Ks * nf)
    Mprime = (K + 4 * G / 3) / ((K + 4 * G / 3) / Mb + alpha ** 2)
    cv = kf * Mprime

    if 1 - 3 * cv * dt / (h2 ** 2) < 0:
        ef = 0
        ef_old = 0
    else:
        ef = (1 / Mprime) * (1 - 3 * cv * dt / (h1 ** 2)) * (1 + np.tanh(2 - 12 * cv * dt / (h1 ** 2))) * ((h1/h2)**(particlePerCell))
        ef_old = (1 / Mprime) * (1 - 3 * cv * dt / (h1 ** 2)) * (1 + np.tanh(2 - 12 * cv * dt / (h1 ** 2)))

    if p == 0:
        xl = [xp[p] + vp[p] * (1 / 3), xp[p] + vp[p] * (2 / 3)]
    elif p == particleNum-1:
        xl = [xp[p] - vp[p] * (1 / 3), xp[p] - vp[p] * (2 / 3)]
    else:
        xl = [xp[p] - vp[p] * (1 / 4), xp[p] + vp[p] * (1 / 4)]
    vl = vp[p] / 2

    phiBar = np.zeros(nodeNum)
    phiU_L,_ = GetRK(xI, dx, xl, supportNorm)
    for l in range(2):
        phiBar += (1 / vp[p]) * phiU_L[:,l] * vl

    return phiBar, ef, ef_old

def nsni(xI,dx,xp,supportNorm,nodeNum,particleNum,domainheight):
    xpp=[]
    xpp.append(0)
    for p in range(particleNum-1):
        xpp.append((xp[p]+xp[p+1])/2)
    xpp.append(domainheight)
    xpp=np.array(xpp)

    MoI = np.zeros(particleNum)
    particleNum1 = len(xpp)

    w = np.zeros((nodeNum, particleNum1))  # Initialize vector for kernel function weights
    dphi_NS = np.zeros((nodeNum, particleNum1))  # Initialize vector for RK shape function gradient
    support = supportNorm * dx           # Define support size for this discretization

    for idx in range(particleNum1):
        M = np.zeros((2, 2))                     # Initialize moment matrix
        for I in range(nodeNum):
            z = abs(xpp[idx] - xI[I]) / support  # Normalized distance between the particle to each node

            if 0 <= z and z <= 0.5:  # Define cubic b-spline kernel function weights
                w[I, idx] = 2/3 - 4*z**2 + 4*z**3
            elif 0.5 <= z and z < 1:
                w[I, idx] = 4/3 - 4*z + 4*z**2 - (4/3)*z**3
            else:
                w[I, idx] = 0

            PxI = np.array([1, xI[I] - xpp[idx]])  # Define P(xI - xp)
            if w[I, idx] != 0:  # If the kernel function weight is not zero
                M = M + w[I, idx] * np.outer(PxI, PxI)  # Define the moment matrix
        for J in range(nodeNum):  # Define RK approximation
            PxI = np.array([1, xI[J] - xpp[idx]])
            PlocGrad = np.array([0, 1])
            dphi_NS[J, idx] = w[J, idx] * np.dot(PlocGrad, np.linalg.solve(M, PxI))

    ddphi_NS = np.zeros((nodeNum,particleNum))
    for p in range(particleNum):
        L1 = xp[p] - xpp[p]
        L2 = xpp[p+1] - xp[p]
        length = L1 + L2
        difference = abs(L2 - L1) / 2
        MoI[p] = length**3 / 12 + difference**2
        ddphi_NS[:,p] = dphi_NS[:,p] * -1 / length + dphi_NS[:,p+1] * 1 / length

    return ddphi_NS,MoI

def main():
    # technique
    stabilization = True
    stabilization_coeff = 1
    VC = True
    NSNI = True
    randomParticle = False

    # material parameters
    E = 1e3 # Young's modulus
    nu = 0.0 # Poission's ratio
    rho = 1
    kh = 1e-5
    gammaf = 10e3
    nf = 0.2 # porosity
    Kf = 2.2e9
    domainHeight = 10
    dt = 1
    simulationTime = 1
    betaNorm = 1000
    supportNorm = 1.5
    g = 0e0
    traction = -1e1
    dirichletBC = 0
    tractionBC = domainHeight

    # discretization settings
    particlePerCell_array = [4]
    nodeNum_array = [21]
    L2_norm = np.zeros((len(particlePerCell_array),2))

    for idx_node in range(len(nodeNum_array)):
        for idx_PPC in range(len(particlePerCell_array)):
            nodeNum = nodeNum_array[idx_node]
            particlePerCell = particlePerCell_array[idx_PPC]
            particleNum, dx, dp, xp, vp, xI, uI, pI, n, timeStepNum = initialize(particlePerCell, nodeNum, domainHeight, dt, simulationTime, randomParticle)
            kf, Ks, alpha, Q_inv, K, G = initialize_terzaghi(E, nu, kh, gammaf, nf, Kf)
            uI_FPP = np.copy(uI)
            pI_FPP = np.copy(pI)
            uI_FPP_old = np.copy(uI)
            pI_FPP_old = np.copy(pI)
            beta = betaNorm * E / dx
            zeroMatrix = np.zeros((nodeNum,nodeNum))
            Kuu = np.copy(zeroMatrix)
            # Kuu_NS = np.copy(zeroMatrix)  # NSNI
            Kup = np.copy(zeroMatrix)
            KppS = np.copy(zeroMatrix)
            KppH = np.copy(zeroMatrix)
            S = np.copy(zeroMatrix)
            S_old = np.copy(zeroMatrix)
            F = np.zeros(nodeNum)
            Q = np.copy(F)

            for i in range(int(simulationTime / dt)):
                phiU, dphiU = GetRK(xI, dx, xp, supportNorm)
                phiP, dphiP = GetRK(xI, dx, xp, supportNorm)
                phiU_VC, dphiU_VC = GetRK(xI, dx, xp, supportNorm)
                _, dphiP_VC = GetRK(xI, dx, xp, supportNorm)

                if VC:
                    ksi, eps = get_vc(phiU_VC, dphiU_VC, nodeNum, vp, n)
                checkVC = 0

                if NSNI:
                    ddphi, MoI = nsni(xI,dx,xp,supportNorm,nodeNum,particleNum,domainHeight)

                for p in range(particleNum):
                    if VC:
                        withinSupport = phiU_VC[:, p] > eps
                        dphiU_VC[:, p] = dphiU_VC[:, p] + ksi * withinSupport
                        dphiP_VC[:, p] = dphiP_VC[:, p] + ksi * withinSupport
                    checkVC = checkVC + np.dot(dphiU_VC[:, p], vp[p]) - np.dot(phiU_VC[:, p], n[p]) * 1

                    Kuu = Kuu + E * np.outer(dphiU_VC[:, p], dphiU[:, p]) * vp[p]
                    # Kuu_NS = Kuu_NS + E * np.outer(dphiU_VC[:, p], dphiU[:, p]) * vp[p]
                    Kup = Kup + alpha * np.outer(dphiU_VC[:, p], phiP[:, p]) * vp[p]
                    KppS = KppS + Q_inv * np.outer(phiP[:, p], phiP[:, p]) * vp[p]
                    KppH = KppH + kf * np.outer(dphiP_VC[:, p], dphiP[:, p]) * vp[p]
                    F = F + phiU[:, p] * rho * g * vp[p]
                    if NSNI:
                        # Kuu_NS = Kuu_NS + E * np.outer(ddphi[:,p],ddphi[:,p]) * MoI[p]
                        Kuu = Kuu + E * np.outer(ddphi[:,p],ddphi[:,p]) * MoI[p]

                    if xp[p] == tractionBC:
                        F = F + phiU[:, p] * traction * 1

                    if stabilization:
                        phiBar, ef ,ef_old = pressure_stabilize(dx, dp, Ks, Kf, kf, alpha, nf, K, G, dt, p, xp, vp, xI, nodeNum, particleNum, supportNorm,particlePerCell)
                        S = S + stabilization_coeff * ef * np.outer(phiP[:, p] - phiBar,phiP[:, p] - phiBar) * vp[p]
                        S_old = S_old + stabilization_coeff * ef_old * np.outer(phiP[:, p] - phiBar,phiP[:, p] - phiBar) * vp[p]

                J = np.block([[Kuu, -Kup], [-Kup.T, (-KppS - dt * KppH)]])
                J_FPP = np.block([[Kuu, -Kup], [-Kup.T, (-KppS - dt * KppH - S)]])
                J_FPP_old = np.block([[Kuu, -Kup], [-Kup.T, (-KppS - dt * KppH - S_old)]])
                Ext = np.concatenate([F, -dt * Q])
                R = np.block([[zeroMatrix, zeroMatrix], [-Kup.T, (-KppS)]])
                R_FPP = np.block([[zeroMatrix, zeroMatrix], [-Kup.T, (-KppS - S)]])
                R_FPP_old = np.block([[zeroMatrix, zeroMatrix], [-Kup.T, (-KppS - S_old)]])
                J = np.delete(J, 0, axis=0)
                J = np.delete(J, 0, axis=1)
                J = np.delete(J, -1, axis=0)
                J = np.delete(J, -1, axis=1)
                J_FPP = np.delete(J_FPP, 0, axis=0)
                J_FPP = np.delete(J_FPP, 0, axis=1)
                J_FPP = np.delete(J_FPP, -1, axis=0)
                J_FPP = np.delete(J_FPP, -1, axis=1)
                J_FPP_old = np.delete(J_FPP_old, 0, axis=0)
                J_FPP_old = np.delete(J_FPP_old, 0, axis=1)
                J_FPP_old = np.delete(J_FPP_old, -1, axis=0)
                J_FPP_old = np.delete(J_FPP_old, -1, axis=1)
                R = np.delete(R, 0, axis=0)
                R = np.delete(R, 0, axis=1)
                R = np.delete(R, -1, axis=0)
                R = np.delete(R, -1, axis=1)
                R_FPP = np.delete(R_FPP, 0, axis=0)
                R_FPP = np.delete(R_FPP, 0, axis=1)
                R_FPP = np.delete(R_FPP, -1, axis=0)
                R_FPP = np.delete(R_FPP, -1, axis=1)
                R_FPP_old = np.delete(R_FPP_old, 0, axis=0)
                R_FPP_old = np.delete(R_FPP_old, 0, axis=1)
                R_FPP_old = np.delete(R_FPP_old, -1, axis=0)
                R_FPP_old = np.delete(R_FPP_old, -1, axis=1)
                Ext = np.delete(Ext, [0, -1])

                X = np.linalg.solve(J, np.dot(R, np.concatenate([uI[1:], pI[:-1]])) + Ext)
                X_FPP = np.linalg.solve(J_FPP, np.dot(R_FPP, np.concatenate([uI_FPP[1:], pI_FPP[:-1]])) + Ext)
                X_FPP_old = np.linalg.solve(J_FPP_old, np.dot(R_FPP_old, np.concatenate([uI_FPP_old[1:], pI_FPP_old[:-1]])) + Ext)
                uI = np.concatenate([[0], X[:nodeNum - 1]])
                pI = np.concatenate([X[nodeNum - 1:], [0]])
                uI_FPP = np.concatenate([[0], X_FPP[:nodeNum - 1]])
                pI_FPP = np.concatenate([X_FPP[nodeNum - 1:], [0]])
                uI_FPP_old = np.concatenate([[0], X_FPP_old[:nodeNum - 1]])
                pI_FPP_old = np.concatenate([X_FPP_old[nodeNum - 1:], [0]])

            step = 0.005
            xc = np.arange(0, domainHeight + step, step)
            phiUc, dphiUc = GetRK(xI, dx, xc, supportNorm)
            uc = np.dot(uI, phiUc)
            pc = np.dot(pI, phiUc)
            pc_FPP = np.dot(pI_FPP, phiUc)
            pc_FPP_old = np.dot(pI_FPP_old, phiUc)

            Z = xc / domainHeight
            pe = 0
            for m in range(10001):  # 注意迴圈範圍的調整
                M = np.pi * (2 * m + 1) / 2
                pe += (2 / M) * np.sin(M * Z)

    plt.figure(1)
    _step = 150

    plt.plot(pc_FPP / abs(traction), Z, 'r-', linewidth=2)
    plt.scatter(pc_FPP[::_step] / abs(traction), Z[::_step], marker='o', color='r',label='VCRKMPM + FPP')

    plt.plot(pc_FPP_old / abs(traction), Z, 'g-', linewidth=2)
    plt.scatter(pc_FPP_old[::_step] / abs(traction), Z[::_step], marker='s', color='g',label='VCRKMPM + FPP(old_coeff)')

    plt.plot(pc / abs(traction), Z, 'b-', linewidth=2,label='VCRKMPM')
    plt.plot(pe, 1-Z, 'k--', linewidth=2,label='Exact Solution')
    plt.xlabel('p/t')
    plt.ylabel('x/Ly')
    plt.legend(bbox_to_anchor=(0, -0.2),loc=2)
    # plt.legend()
    plt.title('Pressure vs Height')
    plt.xlim([-0.1, 2.1])
    plt.show()

    plt.figure(2,(3.0,1.0))
    # plt.plot(xI[17:] * 0, xI[17:], 'rs', markersize=4, linewidth=2)
    # plt.plot(xp[17*particlePerCell:] * 0, xp[17*particlePerCell:], 'ko', markersize=2)
    plt.plot(xI[17:19],xI[17:19] * 0,'rs', markersize=4, linewidth=2)
    plt.plot( xp[17*particlePerCell:18*particlePerCell],xp[17*particlePerCell:18*particlePerCell] * 0, 'ko', markersize=2)
    plt.xticks([])
    plt.yticks([])
    plt.legend(['Grid Nodes','Material Particles'],bbox_to_anchor=(0.5, -0.05),loc=9,fontsize=6)
    plt.show()

if __name__ == "__main__":
    main()