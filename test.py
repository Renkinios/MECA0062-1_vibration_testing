transport = 5
visiteur  = 28 


nbr_train  = visiteur//transport
rest = visiteur%transport
if rest > transport//2:
    nbr_train +=1
print(nbr_train)