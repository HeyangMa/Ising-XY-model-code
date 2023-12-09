"""
示例的主文件 IXYSA_wolff.jl
Author: Ma HY
Date: 2023-08
"""
# H_{y}=\sum  (A+B\sigma_{i}\sigma_{j})*cos(\theta_{i} - \theta_{j})
# H_{x}=\sum  (B+A\sigma_{i}\sigma_{j})*cos(\theta_{i} - \theta_{j})

#xy initialization
function init_xy(n::Int)
    spin = zeros(Float64, n * n, 1)
    # for i = 1:n*n
    #     spin[i] = rand() * 2 * pi
    # end
    return spin
end

#ising initialization
function init_ising(n::Int)
    spin = ones(Int64, n * n, 1)
    # for i = 1:n*n
    #     if rand() > 0.5
    #         spin[i] = -1
    #     end
    # end
    return spin
end

#neighbor relationship
function neighbor(ll)
    neigh = Array{Int}(undef, ll * ll, 4)
    for s0 = 1:ll*ll
        x0 = mod(s0 - 1, ll)
        y0 = div(s0 - 1, ll)
        x1 = mod(x0 + 1, ll)
        x2 = mod(x0 - 1 + ll, ll)
        y1 = mod(y0 + 1, ll)
        y2 = mod(y0 - 1 + ll, ll)
        neigh[s0, 1] = 1 + x1 + y0 * ll
        neigh[s0, 2] = 1 + x0 + y1 * ll
        neigh[s0, 3] = 1 + x2 + y0 * ll
        neigh[s0, 4] = 1 + x0 + y2 * ll
    end
    return neigh
end

#------------------------ metropolis ------------------------
function Jb_deEupdate(ising, xy, nbor, n, beta, A, B)
    for i = 1:n
        if rand() < 0.5
            #x axis
            cen = rand(1:n)
            ising_n = reshape(copy(ising), n, n)'
            ising_n[cen, :] = -ising_n[cen, :]
            ising_n = reshape(ising_n', n * n, 1)
            E_old = calculate_dE(ising, xy, nbor, n, A, B)
            E_new = calculate_dE(ising_n, xy, nbor, n, A, B)
            deE = E_new - E_old
            if rand() < exp(-beta * deE)
                ising = ising_n
            end
        else
            #y axis
            cen = rand(1:n)
            ising_n = reshape(copy(ising), n, n)'
            xy_n = reshape(copy(xy), n, n)'
            ising_n[:, cen] = -ising_n[:, cen]
            xy_n[:, cen] = xy_n[:, cen] .+ pi
            ising_n = reshape(ising_n', n * n, 1)
            xy_n = reshape(xy_n', n * n, 1)
            E_old = calculate_dE(ising, xy, nbor, n, A, B)
            E_new = calculate_dE(ising_n, xy_n, nbor, n, A, B)
            deE = E_new - E_old

            if rand() < exp(-beta * deE)
                ising = ising_n
                xy = xy_n
            end
        end
    end
    # map to (0,2pi)
    for cen = 1:n^2
        while xy[cen] > 2 * pi || xy[cen] < 0
            if xy[cen] > 2 * pi
                xy[cen] = xy[cen] - 2 * pi
            end
            if xy[cen] < 0
                xy[cen] = xy[cen] + 2 * pi
            end
        end
    end
    return xy, ising
end

#calculate dE
function calculate_dE(ising, xy, nbor, n, A, B)
    #energy
    energy = 0.0
    for j = 1:n*n
        energy = energy - ((B + A * ising[j] * ising[nbor[j, 1]]) * cos(xy[j] - xy[nbor[j, 1]]))
        energy = energy - ((A + B * ising[j] * ising[nbor[j, 2]]) * cos(xy[j] - xy[nbor[j, 2]]))
    end
    return energy
end

#------------------------ metropolis ------------------------
function metro_update(ising, xy, nbor, n, beta, A, B)
    for i = 1:n^2
        #pick a site as starting point
        cen = rand(1:n*n)
        #choose a random vector
        alpha = pi * rand()
        #new xy spin
        theta = 2 * alpha - xy[cen]
        while theta > 2 * pi || theta < 0
            if theta > 2 * pi
                theta = theta - 2 * pi
            end
            if theta < 0
                theta = theta + 2 * pi
            end
        end
        E_old = 0
        E_old = E_old - ((B + A * ising[cen] * ising[nbor[cen, 1]]) * cos(xy[cen] - xy[nbor[cen, 1]]))
        E_old = E_old - ((B + A * ising[cen] * ising[nbor[cen, 3]]) * cos(xy[cen] - xy[nbor[cen, 3]]))
        E_old = E_old - ((A + B * ising[cen] * ising[nbor[cen, 2]]) * cos(xy[cen] - xy[nbor[cen, 2]]))
        E_old = E_old - ((A + B * ising[cen] * ising[nbor[cen, 4]]) * cos(xy[cen] - xy[nbor[cen, 4]]))

        E_new = 0
        E_new = E_new - ((B + A * ising[cen] * ising[nbor[cen, 1]]) * cos(theta - xy[nbor[cen, 1]]))
        E_new = E_new - ((B + A * ising[cen] * ising[nbor[cen, 3]]) * cos(theta - xy[nbor[cen, 3]]))
        E_new = E_new - ((A + B * ising[cen] * ising[nbor[cen, 2]]) * cos(theta - xy[nbor[cen, 2]]))
        E_new = E_new - ((A + B * ising[cen] * ising[nbor[cen, 4]]) * cos(theta - xy[nbor[cen, 4]]))

        #P=e^(-\beta δ{E})  therefore,
        deltaE = E_new - E_old
        probability = exp(-1 * beta * deltaE)
        if rand() < probability
            xy[cen] = theta
        end

        #----------------------------------------------------------

        #pick a site as starting point
        cen = rand(1:n*n)
        E_old = 0
        E_old = E_old - ((B + A * ising[cen] * ising[nbor[cen, 1]]) * cos(xy[cen] - xy[nbor[cen, 1]]))
        E_old = E_old - ((B + A * ising[cen] * ising[nbor[cen, 3]]) * cos(xy[cen] - xy[nbor[cen, 3]]))
        E_old = E_old - ((A + B * ising[cen] * ising[nbor[cen, 2]]) * cos(xy[cen] - xy[nbor[cen, 2]]))
        E_old = E_old - ((A + B * ising[cen] * ising[nbor[cen, 4]]) * cos(xy[cen] - xy[nbor[cen, 4]]))

        E_new = 0
        E_new = E_new - ((B - A * ising[cen] * ising[nbor[cen, 1]]) * cos(xy[cen] - xy[nbor[cen, 1]]))
        E_new = E_new - ((B - A * ising[cen] * ising[nbor[cen, 3]]) * cos(xy[cen] - xy[nbor[cen, 3]]))
        E_new = E_new - ((A - B * ising[cen] * ising[nbor[cen, 2]]) * cos(xy[cen] - xy[nbor[cen, 2]]))
        E_new = E_new - ((A - B * ising[cen] * ising[nbor[cen, 4]]) * cos(xy[cen] - xy[nbor[cen, 4]]))

        #P=e^(-\beta δ{E})  therefore,
        deltaE = E_new - E_old
        probability = exp(-1 * beta * deltaE)
        if rand() < probability
            ising[cen] = -ising[cen]
        end
    end
    return xy, ising
end

#------------------------ Wolff xy --------------------------
function Wolff_update_xy(ising, xy, nbor, n, beta, A, B)
    xy_new = copy(xy)
    #choose a random vector
    alpha = pi * rand()
    #pick a site as starting point
    cen = rand(1:n*n)
    cluster = zeros(Int, n * n, 1)
    flag = zeros(Int, n * n, 1)
    #表示把current_site放入了需要查找的队列
    flag[1] = cen
    xy_new[cen] = 2 * alpha - xy_new[cen]
    while xy_new[cen] > 2 * pi || xy_new[cen] < 0
        if xy_new[cen] > 2 * pi
            xy_new[cen] = xy_new[cen] - 2 * pi
        end
        if xy_new[cen] < 0
            xy_new[cen] = xy_new[cen] + 2 * pi
        end
    end
    #修改值为1，表示该格点意见被选择过
    cluster[cen] = 1
    #表示待查找的点的个数+1，如果为1就结束
    counter = 2
    #开始查找
    while counter > 1
        #从最后一个被加入的点开始查找
        counter = counter - 1
        cen = flag[counter]
        flag[counter] = 0
        for i = 1:4
            #查找每一个近邻
            next_site = nbor[cen, i]
            #需要满足条件
            # H_{y}=\sum  (A+B\sigma_{i}\sigma_{j})*cos(\theta_{i} - \theta_{j}) + C * \sigma_{i}\sigma_{j}
            # H_{x}=\sum  (B+A\sigma_{i}\sigma_{j})*cos(\theta_{i} - \theta_{j}) + C * \sigma_{i}\sigma_{j}
            if i == 1 || i == 3
                E_old = -((B + A * ising[cen] * ising[nbor[cen, i]]) * (cos(xy[cen] - xy[nbor[cen, i]])))
                E_new = -((B + A * ising[cen] * ising[nbor[cen, i]]) * (cos(xy_new[cen] - xy[nbor[cen, i]])))
            else
                E_old = -((A + B * ising[cen] * ising[nbor[cen, i]]) * (cos(xy[cen] - xy[nbor[cen, i]])))
                E_new = -((A + B * ising[cen] * ising[nbor[cen, i]]) * (cos(xy_new[cen] - xy[nbor[cen, i]])))
            end
            deltaE = E_new - E_old
            #P=e^(-\beta δ{E})  therefore,
            probability = 1 - exp(-1 * beta * deltaE)
            if rand() < probability && cluster[next_site] == 0
                xy_new[next_site] = 2 * alpha - xy_new[next_site]
                while xy_new[next_site] > 2 * pi || xy_new[next_site] < 0
                    if xy_new[next_site] > 2 * pi
                        xy_new[next_site] = xy_new[next_site] - 2 * pi
                    end
                    if xy_new[next_site] < 0
                        xy_new[next_site] = xy_new[next_site] + 2 * pi
                    end
                end
                #将满足条件的点放入待查找的队列中
                flag[counter] = next_site
                cluster[next_site] = 1
                counter = counter + 1
            end
        end
    end
    # map to (0,2pi)
    for cen = 1:n^2
        while xy_new[cen] > 2 * pi || xy_new[cen] < 0
            if xy_new[cen] > 2 * pi
                xy_new[cen] = xy_new[cen] - 2 * pi
            end
            if xy_new[cen] < 0
                xy_new[cen] = xy_new[cen] + 2 * pi
            end
        end
    end
    return xy_new
end

#------------------------ Wolff ising -----------------------
function Wolff_update_ising(ising, xy, nbor, n, beta, A, B)
    #pick a site as starting point
    cen = rand(1:n*n)
    cluster = zeros(Int, n * n, 1)
    flag = zeros(Int, n * n, 1)
    #表示把current_site放入了需要查找的队列
    flag[1] = cen
    #修改值为1，表示该格点意见被选择过
    cluster[cen] = 1
    #表示待查找的点的个数+1，如果为1就结束
    counter = 2
    #开始查找
    while counter > 1
        #从最后一个被加入的点开始查找
        counter = counter - 1
        cen = flag[counter]
        flag[counter] = 0
        for i = 1:4
            #查找每一个近邻
            next_site = nbor[cen, i]
            #需要满足条件
            # H_{y}=\sum  (A+B\sigma_{i}\sigma_{j})*cos(\theta_{i} - \theta_{j}) + C * \sigma_{i}\sigma_{j}
            # H_{x}=\sum  (B+A\sigma_{i}\sigma_{j})*cos(\theta_{i} - \theta_{j}) + C * \sigma_{i}\sigma_{j}
            if i == 1 || i == 3
                E_old = -((B + A * ising[cen] * ising[nbor[cen, i]]) * (cos(xy[cen] - xy[nbor[cen, i]])))
                E_new = -((B - A * ising[cen] * ising[nbor[cen, i]]) * (cos(xy[cen] - xy[nbor[cen, i]])))
            else
                E_old = -((A + B * ising[cen] * ising[nbor[cen, i]]) * (cos(xy[cen] - xy[nbor[cen, i]])))
                E_new = -((A - B * ising[cen] * ising[nbor[cen, i]]) * (cos(xy[cen] - xy[nbor[cen, i]])))
            end
            deltaE = E_new - E_old
            #P=e^(-\beta δ{E})  therefore,
            probability = 1 - exp(-1 * beta * deltaE)
            if rand() < probability && cluster[next_site] == 0
                #将满足条件的点放入待查找的队列中
                flag[counter] = next_site
                cluster[next_site] = 1
                counter = counter + 1
            end
        end
    end
    for i = 1:n*n
        if cluster[i] == 1
            #翻转ising构型
            ising[i] = -ising[i]
        end
    end
    return ising
end

#------------------------ Swendsen-wang ising ---------------
function swendsen_wang_ising(ising, xy, nbor, n, beta, A, B)
    # one Monte Carlo step
    bonds = configure_bonds_ising(ising, xy, nbor, n, beta, A, B)
    cluster = search_cluster(bonds, nbor, n)
    ising = flip_spin_ising(cluster, ising)
    return ising
end

#------------------------ Swendsen-wang xy ------------------
function swendsen_wang_xy(ising, xy, nbor, n, beta, A, B)
    # one Monte Carlo step
    alpha = 2 * pi * rand()
    bonds = configure_bonds_xy(ising, xy, nbor, n, beta, A, B, alpha)
    cluster = search_cluster(bonds, nbor, n)
    xy = flip_spin_xy(cluster, xy, alpha)
    # map to (0,2pi)
    for cen = 1:n^2
        while xy[cen] > 2 * pi || xy[cen] < 0
            if xy[cen] > 2 * pi
                xy[cen] = xy[cen] - 2 * pi
            end
            if xy[cen] < 0
                xy[cen] = xy[cen] + 2 * pi
            end
        end
    end
    return xy
end

#[c,d]=project_t(a,b)
function project_t(a, b)
    theta = a
    theta_proj = b
    # 计算角向量和投影方向向量
    vec_angle = [cos(theta), sin(theta)]
    proj_dir = [cos(theta_proj), sin(theta_proj)]
    # 计算投影值
    proj_value = vec_angle[1] * proj_dir[1] + vec_angle[2] * proj_dir[2]

    ## --------------给定两个角度和方向---------------------
    theta1 = a
    theta2 = b

    # 计算两个方向向量
    dir1 = [cos(theta1), sin(theta1)]
    dir2 = [cos(theta2), sin(theta2)]

    # 计算反射向量
    reflection_dir = dir1 - (2 * (dir1[1] * dir2[1] + dir1[2] * dir2[2])) * dir2

    # 计算反射角
    theta_refl = atan(reflection_dir[2], reflection_dir[1])
    if theta_refl < 0
        theta_refl = 2 * pi + theta_refl # 保证反射角度在0到360之间
    end
    angle_refl = theta_refl * 180 / pi
    d = angle_refl * pi / 180

    proj_value, d
end

#configure bonds ising
function configure_bonds_ising(ising, xy, nbor, n, beta, A, B)
    bonds = zeros(Int, n * n, 2)
    #1:right  2:up  3:left  4:down
    #place bonds
    # H_{y}=\sum  (A+B\sigma_{i}\sigma_{j})*cos(\theta_{i} - \theta_{j}) + C * \sigma_{i}\sigma_{j}
    # H_{x}=\sum  (B+A\sigma_{i}\sigma_{j})*cos(\theta_{i} - \theta_{j}) + C * \sigma_{i}\sigma_{j}
    for i = 1:n*n
        if rand() < (1 - exp(-2 * (A * cos(xy[i] - xy[nbor[i, 1]])) * beta * ising[i] * ising[nbor[i, 1]]))
            bonds[i, 1] = 1
        end
        if rand() < (1 - exp(-2 * (B * cos(xy[i] - xy[nbor[i, 2]])) * beta * ising[i] * ising[nbor[i, 2]]))
            bonds[i, 2] = 1
        end
    end
    #----End----
    return bonds
end

#configure bonds xy
function configure_bonds_xy(ising, xy, nbor, n, beta, A, B, alpha)
    bonds = zeros(Int, n * n, 2)
    #计算投影
    xy_pro = zeros(Float64, n * n, 1)
    for j = 1:n*n
        xy_pro[j], ~ = project_t(xy[j], alpha)
    end
    #place bonds
    # H_{y}=\sum  (A+B\sigma_{i}\sigma_{j})*cos(\theta_{i} - \theta_{j}) + C * \sigma_{i}\sigma_{j}
    # H_{x}=\sum  (B+A\sigma_{i}\sigma_{j})*cos(\theta_{i} - \theta_{j}) + C * \sigma_{i}\sigma_{j}
    for j = 1:n*n
        if rand() < max(0, 1 - exp(-2 * (B + A * ising[j] * ising[nbor[j, 1]]) * beta * xy_pro[j] * xy_pro[nbor[j, 1]]))
            bonds[j, 1] = 1
        end
        if rand() < max(0, 1 - exp(-2 * (A + B * ising[j] * ising[nbor[j, 2]]) * beta * xy_pro[j] * xy_pro[nbor[j, 2]]))
            bonds[j, 2] = 1
        end
    end
    #----End----
    return bonds
end

#search nbor site
function searchnbor(number, nbor, cen)
    if number == 1
        sitenbor = nbor[cen, 1]
    elseif number == 2
        sitenbor = nbor[cen, 2]
        # elseif number == 3
        #     sitenbor = nbor[nbor[cen, 2], 1]
        # elseif number == 4
        #     sitenbor = nbor[nbor[cen, 2], 3]
        # elseif number==5
        #     sitenbor=nbor(nbor(nbor(cen,3),3),2);
        # elseif number==6
        #     sitenbor=nbor(nbor(nbor(cen,3),2),2);
        # elseif number==7
        #     sitenbor=nbor(nbor(nbor(cen,1),2),2);
        # elseif number==8
        #     sitenbor=nbor(nbor(nbor(cen,1),1),2);
        # elseif number==9
        #     sitenbor=nbor(nbor(cen,2),2);
        # elseif number==10
        #     sitenbor=nbor(nbor(cen,3),3);
    end
    return sitenbor
end

#find root label
function find_root(m, label)
    y = m
    while label[y] != y
        y = label[y]
    end
    y, label
end

#union bonds
function union(notzerolabellist, label)
    var = Array{Int}(undef, length(notzerolabellist), 1)
    for k = 1:length(notzerolabellist)
        aaa = find_root(notzerolabellist[k], label)
        var[k] = aaa[1]
        label = aaa[2]
    end
    a::Int64 = findmin(var)[1]
    for k = 1:length(notzerolabellist)
        label[var[k]] = a
    end
    m = a
    label, m
end

#apply the H-K algorithm and search cluster with PBC
function search_cluster(bonds, nbor, n)
    cluster = zeros(Int64, n * n, 1) #cluster label
    label = Array{Int64}(1:n*n) #root label
    largest_label = 0
    for j = 1:n*n
        #1:right  2:up 
        bondlist = [bonds[j, 1], bonds[j, 2]]
        bondsum = sum(bondlist)
        if cluster[j] == 0 #current site unlabeled
            if bondsum == 0 #no bond connection
                largest_label = largest_label + 1
                cluster[j] = largest_label
            elseif bondsum == 1 ##one bond connection
                number = findall(==(1), bondlist)
                sitenbor = searchnbor(number[1], nbor, j)
                if cluster[sitenbor] == 0
                    largest_label = largest_label + 1
                    cluster[j] = largest_label
                    cluster[sitenbor] = largest_label
                else
                    aaa = find_root(cluster[sitenbor], label)
                    cluster[j] = aaa[1]
                    label = aaa[2]
                end
            elseif bondsum > 1 #more than one bond connection
                numberlist = findall(==(1), bondlist)
                #use the following array to memory
                varlist = zeros(Int, length(numberlist), 1)   #memory cluster label
                sitenbor1 = zeros(Int, length(numberlist), 1) #memory site  label
                for k = 1:length(numberlist)
                    sitenbor1[k] = searchnbor(numberlist[k], nbor, j)
                    varlist[k] = cluster[sitenbor1[k]]
                end
                labellist = varlist[findall(!=(0), varlist)]
                if labellist == []
                    largest_label = largest_label + 1
                    for k = 1:length(numberlist)
                        cluster[sitenbor1[k]] = largest_label
                    end
                    cluster[j] = largest_label
                else
                    aaa = union(labellist, label)
                    label = aaa[1]
                    minlabel = aaa[2]
                    for k = 1:length(numberlist)
                        cluster[sitenbor1[k]] = minlabel
                    end
                    cluster[j] = minlabel
                end
            end
        else #The current site has a non-zeros label
            if bondsum == 0 #no bond
                continue
            elseif bondsum > 0 #more than one bond (current site bring one)
                numberlist = findall(==(1), bondlist)
                #use the following array to memory
                varlist = zeros(Int, length(numberlist), 1)   #memory cluster label
                sitenbor1 = zeros(Int, length(numberlist), 1) #memory site label
                for k = 1:length(numberlist)
                    sitenbor1[k] = searchnbor(numberlist[k], nbor, j)
                    varlist[k] = cluster[sitenbor1[k]]
                end
                labellist = varlist[findall(!=(0), varlist)]
                if labellist == []
                    for k = 1:length(numberlist)
                        aaa = find_root(cluster[j], label)
                        a = aaa[1]
                        label = aaa[2]
                        cluster[sitenbor1[k]] = a
                    end
                else
                    aaa = union(labellist, label)
                    label = aaa[1]
                    minlabel = aaa[2]
                    aaa = find_root(cluster[j], label)
                    a = aaa[1]
                    label = aaa[2]
                    sminlabel = findmin([minlabel, a])[1]
                    label[minlabel] = sminlabel
                    label[a] = sminlabel
                    for k = 1:length(numberlist)
                        cluster[sitenbor1[k]] = minlabel
                    end
                    cluster[j] = minlabel
                end
            end
        end
    end
    for j = 1:n*n
        aaa = find_root(cluster[j], label)
        cluster[j] = aaa[1]
        label = aaa[2]
    end
    return cluster
end

#flip cluster's spins ising
function flip_spin_ising(cluster, ising)
    for i = 1:findmax(cluster)[1]
        if rand() > 0.5 && findall(==(i), cluster) != []  #成功翻转 && 有这一标签的集团
            n = findall(==(i), cluster)
            for j = 1:length(n)
                ising[n[j]] = -ising[n[j]]
            end
        end
    end
    return ising
end

#flip cluster's spins xy
function flip_spin_xy(cluster, xy, alpha)
    for i = 1:findmax(cluster)[1]
        if rand() > 0.5 && findall(==(i), cluster) != []  #成功翻转 && 有这一标签的集团
            n = findall(==(i), cluster)

            for j = 1:length(n)
                #xy
                ~, xy[n[j]] = project_t(xy[n[j]], alpha)
            end
        end
    end
    return xy
end

#calculate other phys quantities
function Jackknife(data)
    ave_data = zeros(Float64, length(data), 1)
    for i = 1:length(data)
        a = 0
        for j = 1:length(data)
            if j != i
                a = a + data[j]
            end
        end
        ave_data[i] = a / (length(data) - 1)
    end
    expect = sum(data) / length(data) - (length(data) - 1) * (sum(ave_data) / length(data) - sum(data) / length(data))
    b = ave_data - ones(Float64, length(data), 1) * (sum(ave_data) / length(data))
    delta_expect = sqrt(((length(data) - 1) / length(data)) * sum(b .^ 2))
    expect, delta_expect
end

#sample variance
function my_var(data)
    mean_data = sum(data) / length(data)
    s = 0
    for i = 1:length(data)
        s = s + (data[i] - mean_data)^2
    end
    s = s / (length(data) - 1)
    return s
end

function count_integer_vortex(xy, nbor, n)
    # map to (-pi,+pi)
    for cen = 1:n^2
        while xy[cen] > pi || xy[cen] < -pi
            if xy[cen] > pi
                xy[cen] = xy[cen] - 2 * pi
            end
            if xy[cen] < -pi
                xy[cen] = xy[cen] + 2 * pi
            end
        end
    end
    #---------count integer vortex---------
    theta = zeros(Float64, 4, 1)
    v = zeros(Float64, 4, 1)
    vortex = 0
    for i = 1:n^2
        theta[1] = xy[nbor[nbor[i, 1], 2]]
        theta[2] = xy[nbor[i, 2]]
        theta[3] = xy[i]
        theta[4] = xy[nbor[i, 1]]
        v[1] = theta[2] - theta[1]
        v[2] = theta[3] - theta[2]
        v[3] = theta[4] - theta[3]
        v[4] = theta[1] - theta[4]
        for k = 1:4
            while v[k] <= -pi || v[k] > pi
                if v[k] <= -pi
                    v[k] = v[k] + 2 * pi
                end
                if v[k] > pi
                    v[k] = v[k] - 2 * pi
                end
            end
        end
        if abs(sum(v) - 2 * pi) <= 1e-5
            vortex = vortex + 1
        end
        if abs(sum(v) + 2 * pi) <= 1e-5
            vortex = vortex + 1
        end
    end
    return vortex
end

function count_half_vortex(xy, nbor, n)
    # map to (-pi,+pi)
    for cen = 1:n^2
        while xy[cen] > pi || xy[cen] < -pi
            if xy[cen] > pi
                xy[cen] = xy[cen] - 2 * pi
            end
            if xy[cen] < -pi
                xy[cen] = xy[cen] + 2 * pi
            end
        end
    end
    #---------count half vortex---------
    theta = zeros(Float64, 4, 1)
    v = zeros(Float64, 4, 1)
    vortex = 0
    error = 0.01
    for i = 1:n^2
        theta[1] = xy[nbor[nbor[i, 1], 2]]
        theta[2] = xy[nbor[i, 2]]
        theta[3] = xy[i]
        theta[4] = xy[nbor[i, 1]]
        v[1] = theta[2] - theta[1]
        v[2] = theta[3] - theta[2]
        v[3] = theta[4] - theta[3]
        v[4] = theta[1] - theta[4]
        for k = 1:4
            while v[k] <= -pi / 2 || v[k] > pi / 2
                if v[k] <= -pi / 2
                    v[k] = v[k] + 2 * pi / 2
                end
                if v[k] > pi / 2
                    v[k] = v[k] - 2 * pi / 2
                end
            end
        end
        #----判断条件1
        if abs(sum(v) - pi) <= 1e-5
            vortex = vortex + 1
        end
        if abs(sum(v) + pi) <= 1e-5
            vortex = vortex + 1
        end
    end
    return vortex
end

function measure_cluster_size(ising, n, nbor)
    bonds = zeros(Int, n * n, 2)
    for i = 1:n*n
        if ising[i] == ising[nbor[i, 1]]
            bonds[i, 1] = 1
        end
        if ising[i] == ising[nbor[i, 2]]
            bonds[i, 2] = 1
        end
    end
    cluster = search_cluster(bonds, nbor, n)
    #统计集团大小
    # 统计每个数值的个数
    counts = StatsBase.countmap(cluster)
    # 按降序排列
    sorted_counts = sort(collect(counts), by=x -> x[2], rev=true)
    # 输出结果到浮点型一维数组
    result = Float64[]
    for pair in sorted_counts
        push!(result, pair[2])  # 元素个数
    end
    return result
end

#calculate physical quantities in equilibrium
function calculate(ising, xy, nbor, n, A, B)
    # H_{y}=\sum  (A+B\sigma_{i}\sigma_{j})*cos(\theta_{i} - \theta_{j})
    # H_{x}=\sum  (B+A\sigma_{i}\sigma_{j})*cos(\theta_{i} - \theta_{j})

    #energy
    energy = 0.0
    for j = 1:n*n
        energy = energy - ((B + A * ising[j] * ising[nbor[j, 1]]) * cos(xy[j] - xy[nbor[j, 1]]))
        energy = energy - ((A + B * ising[j] * ising[nbor[j, 2]]) * cos(xy[j] - xy[nbor[j, 2]]))
    end

    #energy_sq
    energy_sq = energy^2

    #mag
    m_ising = abs(sum(ising))
    m_ising_sq = m_ising^2
    m_ising_sq4 = m_ising^4


    #另一个xy
    xy2 = zeros(Float64, n * n, 1)
    for j = 1:n*n
        xy2[j] = xy[j] + ising[j] * (pi / 2)
    end

    #xy_double
    xy_double = 2 * xy
    for i = 1:n^2
        while xy_double[i] > 2 * pi || xy_double[i] < 0
            if xy_double[i] > 2 * pi
                xy_double[i] = xy_double[i] - 2 * pi
            end
            if xy_double[i] < 0
                xy_double[i] = xy_double[i] + 2 * pi
            end
        end
    end

    # theta1 + theta2
    xyxy = xy2 + xy
    for i = 1:n^2
        while xyxy[i] > 2 * pi || xyxy[i] < 0
            if xyxy[i] > 2 * pi
                xyxy[i] = xyxy[i] - 2 * pi
            end
            if xyxy[i] < 0
                xyxy[i] = xyxy[i] + 2 * pi
            end
        end
    end

    # 2*(theta1 + theta2)
    xyxy_double = 2 * (xy2 + xy)
    for i = 1:n^2
        while xyxy_double[i] > 2 * pi || xyxy_double[i] < 0
            if xyxy_double[i] > 2 * pi
                xyxy_double[i] = xyxy_double[i] - 2 * pi
            end
            if xyxy_double[i] < 0
                xyxy_double[i] = xyxy_double[i] + 2 * pi
            end
        end
    end

    # xy_pi
    xy_pi = copy(xyxy)
    for i = 1:n^2
        while xy_pi[i] > 1 * pi || xy_pi[i] < 0
            if xy_pi[i] > 1 * pi
                xy_pi[i] = xy_pi[i] - 1 * pi
            end
            if xy_pi[i] < 0
                xy_pi[i] = xy_pi[i] + 1 * pi
            end
        end
    end

    m_xy = sqrt((sum(cos.(xy)))^2 + (sum(sin.(xy)))^2)                            #theta1
    m_xy_sq = m_xy^2
    m_xy_sq4 = m_xy^4
    m_xy2 = sqrt((sum(cos.(xy_double)))^2 + (sum(sin.(xy_double)))^2)             #theta1 * 2
    m_xy2_sq = m_xy2^2
    m_xy2_sq4 = m_xy2^4
    m_xyxy = sqrt((sum(cos.(xyxy)))^2 + (sum(sin.(xyxy)))^2)                      #theta1 + theta2
    m_xyxy_sq = m_xyxy^2
    m_xyxy_sq4 = m_xyxy^4
    m_xyxy_double = sqrt((sum(cos.(xyxy_double)))^2 + (sum(sin.(xyxy_double)))^2) #(theta1 + theta2) * 2
    m_xyxy_double_sq = m_xyxy_double^2
    m_xyxy_double_sq4 = m_xyxy_double^4
    m_xy_pi = sqrt((sum(cos.(xy_pi)))^2 + (sum(sin.(xy_pi)))^2)                   #m_xy_pi
    m_xy_pi_sq = m_xy_pi^2
    m_xy_pi_sq4 = m_xy_pi^4

    #bonds order
    bond_order_x = 0
    bond_order_y = 0
    for i = 1:n^2
        bond_order_x = bond_order_x + ising[i] * ising[nbor[i, 1]]
        bond_order_y = bond_order_y + ising[i] * ising[nbor[i, 2]]
    end

    #theta1+theta2 : H_x H_y I_x I_y
    H_x = 0
    H_y = 0
    I_x = 0
    I_y = 0
    for j = 1:n^2
        H_x = H_x + ((B + A * ising[j] * ising[nbor[j, 1]]) * (cos(xy[j] - xy[nbor[j, 1]])))
        H_y = H_y + ((A + B * ising[j] * ising[nbor[j, 2]]) * (cos(xy[j] - xy[nbor[j, 2]])))
        I_x = I_x + ((B + A * ising[j] * ising[nbor[j, 1]]) * (sin(xy[j] - xy[nbor[j, 1]])))
        I_y = I_y + ((A + B * ising[j] * ising[nbor[j, 2]]) * (sin(xy[j] - xy[nbor[j, 2]])))
    end
    I_x_sq = I_x^2
    I_y_sq = I_y^2

    # measure percolation
    cluster_size = measure_cluster_size(ising, n, nbor)
    #------deng: S2-------------------------
    S2 = sum(cluster_size[2:end] .^ 2)
    #------size of the largest cluster------
    larg_cluster_size = cluster_size[1]
    larg_cluster_size_sq = larg_cluster_size^2
    #------article: S0S2S4------------------
    S00 = sum(cluster_size .^ 0)
    S22 = sum(cluster_size .^ 2)
    S44 = sum(cluster_size .^ 4)
    S22_sq = S22^2
    q_3S2_S4 = 3 * S22_sq - 2 * S44

    #ising flux
    flux = 0.0
    for j = 1:n*n
        flux = flux + ising[j] * ising[nbor[j, 1]] * ising[nbor[j, 2]] * ising[nbor[nbor[j, 1], 2]]
    end

    #integer vortex
    integer_vortex = count_integer_vortex(xy, nbor, n)

    #half vortex
    half_vortex = count_half_vortex(xy, nbor, n)

    integer_vortex_sq = integer_vortex^2
    half_vortex_sq = half_vortex^2

    result = zeros(Float64, 1, 38)
    result = [energy, energy_sq, m_ising, m_ising_sq, m_ising_sq4, m_xy, m_xy_sq, m_xy_sq4, m_xy2, m_xy2_sq, m_xy2_sq4, m_xyxy, m_xyxy_sq, m_xyxy_sq4, m_xyxy_double, m_xyxy_double_sq, m_xyxy_double_sq4, m_xy_pi, m_xy_pi_sq, m_xy_pi_sq4, bond_order_x, bond_order_y, H_x, H_y, I_x_sq, I_y_sq, S2, flux, larg_cluster_size, S00, S22, q_3S2_S4, larg_cluster_size_sq, S22_sq, integer_vortex, half_vortex, integer_vortex_sq, half_vortex_sq]

    return result
end

#process_bindata
function process_bin(a, n, T)
    #Input : 平均后的物理量(未平均到每个格点)
    #energy     : energy, energy_sq,                                                                 2
    #Ising      : m_ising, m_ising_sq, m_ising_sq4,                                                  3
    #xy         : m_xy, m_xy_sq, m_xy_sq4,
    #             m_xy2, m_xy2_sq, m_xy2_sq4,
    #             m_xyxy, m_xyxy_sq, m_xyxy_sq4,
    #             m_xyxy_double, m_xyxy_double_sq, m_xyxy_double_sq4,
    #             m_xy_pi, m_xy_pi_sq, m_xy_pi_sq4,                                                 15
    #bond order : bond_order_x, bond_order_y,                                                        2
    #rho        : H_x, H_y, I_x_sq, I_y_sq,                                                          4
    #others     : S2, flux, larg_cluster_size, S00, S22, q_3S2_S4, larg_cluster_size_sq, S22_sq,     8
    #vortex     : integer_vortex, half_vortex                                                        2

    #energy & cv
    energy = a[1] / n^2

    cv = (a[2] - a[1]^2) / T^2 / n^2

    #mag ising & ms & Br_ising
    m_ising = a[3] / n^2
    ms_ising = (a[4] - a[3]^2) / T / n^2
    R2_ising = a[5] / a[4]^2
    Br_ising = 3 / 2 * (1 - R2_ising / 3)

    #mag XY & ms & Br xy
    m_xy = a[6] / n^2
    ms_xy = (a[7] - a[6]^2) / T / n^2
    R2_xy = a[8] / a[7]^2
    Br_xy = 2 * (1 - R2_xy / 2)
    #---
    m_xy2 = a[9] / n^2
    ms_xy2 = (a[10] - a[9]^2) / T / n^2
    R2_xy2 = a[11] / a[10]^2
    Br_xy2 = 2 * (1 - R2_xy2 / 2)
    #---
    m_xyxy = a[12] / n^2
    ms_xyxy = (a[13] - a[12]^2) / T / n^2
    R2_xyxy = a[14] / a[13]^2
    Br_xyxy = 2 * (1 - R2_xyxy / 2)
    #---
    m_xyxy_double = a[15] / n^2
    ms_xyxy_double = (a[16] - a[15]^2) / T / n^2
    R2_xyxy2 = a[17] / a[16]^2
    Br_xyxy2 = 2 * (1 - R2_xyxy2 / 2)
    #---
    m_xy_pi = a[18] / n^2
    ms_xy_pi = (a[19] - a[18]^2) / T / n^2
    R2_xy_pi = a[20] / a[19]^2
    Br_xy_pi = 2 * (1 - R2_xy_pi / 2)

    #bonds_order
    bonds_order_x = a[21] / n^2
    bonds_order_y = a[22] / n^2

    #rho
    rho_x = (a[23] - (1 / T) * a[25]) / n^2
    rho_y = (a[24] - (1 / T) * a[26]) / n^2
    rho = (rho_x + rho_y) / 2

    #others
    S2 = a[27] / n^2     #cluster size
    flux = a[28] / n^2   #flux
    #percolation
    larg_cluster_size = a[29] / n^2
    cluster_density = a[30] / n^2
    S22 = a[31] / n^2
    Q1 = (a[33]) / (a[29]^2)  #larg_cluster_size_sq/larg_cluster_size
    Q2 = a[34] / a[32]        #S22_sq/q_3S2_S4

    #integer vortex & half vortex & fluctuation
    integer_vortex = a[35] / n^2
    half_vortex = a[36] / n^2
    flu_integer_vortex = (a[37] - a[35]^2) / n^2
    flu_half_vortex = (a[38] - a[36]^2) / n^2

    energy, cv, m_ising, ms_ising, Br_ising, m_xy, ms_xy, Br_xy, m_xy2, ms_xy2, Br_xy2, m_xyxy, ms_xyxy, Br_xyxy, m_xyxy_double, ms_xyxy_double, Br_xyxy2, m_xy_pi, ms_xy_pi, Br_xy_pi, bonds_order_x, bonds_order_y, rho_x, rho_y, rho, S2, flux, larg_cluster_size, cluster_density, S22, Q1, Q2, integer_vortex, half_vortex, flu_integer_vortex, flu_half_vortex
end

#a certain temperature Monte Carlo simulate
function mcmc(n, T, Thermal, bins, bsteps, A, B, certain, diff, wolff, sw, metro)
    beta = 1 / T
    xy = init_xy(n)
    ising = init_ising(n)
    nbor = neighbor(n)
    for j = 1:Thermal
        #------------ Wolff ------------
        if wolff == 1
            xy = Wolff_update_xy(ising, xy, nbor, n, beta, A, B)
            ising = Wolff_update_ising(ising, xy, nbor, n, beta, A, B)
        end
        #------------  SW --------------
        if sw == 1
            xy = swendsen_wang_xy(ising, xy, nbor, n, beta, A, B)
            ising = swendsen_wang_ising(ising, xy, nbor, n, beta, A, B)
            xy, ising = swendsen_wang_all(ising, xy, nbor, n, beta, A, B)
        end
        #------------ metro ------------
        if metro == 1
            xy, ising = metro_update(ising, xy, nbor, n, beta, A, B)
            xy, ising = Jb_deEupdate(ising, xy, nbor, n, beta, A, B)
        end
    end
    # certain T
    if certain == 1
        write_data(n, T, beta, ising, xy, nbor, bins, bsteps, A, B, wolff, sw, metro)
    end
    # different T
    if diff == 1
        write_data2(n, T, beta, ising, xy, nbor, bins, bsteps, A, B, wolff, sw, metro)
    end
end

#write data at certain T
function write_data(n, T, beta, ising, xy, nbor, bins, bsteps, A, B, wolff, sw, metro)

    for j = 1:bins*bsteps
        #------------ Wolff ------------
        if wolff == 1
            xy = Wolff_update_xy(ising, xy, nbor, n, beta, A, B)
            ising = Wolff_update_ising(ising, xy, nbor, n, beta, A, B)
        end
        #------------  SW --------------
        if sw == 1
            xy = swendsen_wang_xy(ising, xy, nbor, n, beta, A, B)
            ising = swendsen_wang_ising(ising, xy, nbor, n, beta, A, B)
            xy, ising = swendsen_wang_all(ising, xy, nbor, n, beta, A, B)
        end
        #------------ metro ------------
        if metro == 1
            xy, ising = metro_update(ising, xy, nbor, n, beta, A, B)
            xy, ising = Jb_deEupdate(ising, xy, nbor, n, beta, A, B)
        end

        #ising flux
        flux = 0.0
        for j = 1:n*n
            if ising[j] * ising[nbor[j, 1]] * ising[nbor[j, 2]] * ising[nbor[nbor[j, 1], 2]] == -1
                flux = 1
            end
        end

        #vortex
        vortex = count_integer_vortex(xy, nbor, n)

        # if flux == 1 && vortex != 0
        # write configuration
        dizhi1 = string("C://Users//MHY//Desktop//ising.txt")
        dizhi2 = string("C://Users//MHY//Desktop//xy.txt")
        open(dizhi1, "a") do io
            writedlm(io, [ising], ',')
        end
        open(dizhi2, "a") do io
            writedlm(io, [xy], ',')
        end
        # end

    end
end

#write data at different T with error
function write_data2(n, T, beta, ising, xy, nbor, bins, bsteps, A, B, wolff, sw, metro)
    bin = zeros(Float64, bins, 36)
    for j = 1:bins
        bstep = zeros(Float64, 38)
        for i = 1:bsteps
            #------------ Wolff ------------
            if wolff == 1
                xy = Wolff_update_xy(ising, xy, nbor, n, beta, A, B)
                ising = Wolff_update_ising(ising, xy, nbor, n, beta, A, B)
            end
            #------------  SW --------------
            if sw == 1
                xy = swendsen_wang_xy(ising, xy, nbor, n, beta, A, B)
                ising = swendsen_wang_ising(ising, xy, nbor, n, beta, A, B)
                xy, ising = swendsen_wang_all(ising, xy, nbor, n, beta, A, B)
            end
            #------------ metro ------------
            if metro == 1
                xy, ising = metro_update(ising, xy, nbor, n, beta, A, B)
                xy, ising = Jb_deEupdate(ising, xy, nbor, n, beta, A, B)
            end

            #------------ measure ------------
            bstep[:] += calculate(ising, xy, nbor, n, A, B)
        end
        bstep = bstep / bsteps
        #------------ process bindata ------------
        bin[j, 1:36] .= process_bin(bstep, n, T)
    end

    phys = zeros(Float64, size(bin, 2), 2)
    for j = 1:size(bin, 2)
        phys[j, 1], phys[j, 2] = Jackknife(bin[:, j])
    end

    #print data
    phys = vec(transpose(phys))
    dizhi = string("C://Users//MHY//Desktop//tesdata_phyL.txt")
    open(dizhi, "a") do io
        data = vcat([T, n, A, B], phys)
        writedlm(io, [data], ',')
    end
end


using DelimitedFiles
using StatsBase
#--------参数设置--------
n = 10
A = 1
B = 0
T = (0:0.05:1)
Thermal = 5 * 10000              #弛豫步数
bins = 10                       #bins数目
bsteps = 10000                    #每个bin内的步数
#--------模式控制--------
certain = 0
diff = 1
#--------算法控制--------
wolff = 1
sw = 0
metro = 1
#--------------------------------------------------------------------------
@time begin
    for i = 1:length(T)
        mcmc(Int(n), T[i], Thermal, bins, bsteps, A, B, certain, diff, wolff, sw, metro)
    end
end