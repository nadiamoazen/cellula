

class Animal():
    DIET = []  # The list of food that the animal can eat.
    Numb_male =0   ## class attribute
    Numb_female =0 ## class attribute

    # TODO
    def __init__(self, name , gender , fav_food):
        self.name= name
        self.gender = gender
        self.fav_food= fav_food
        array_name.append(self.name)
        array_gender.append(self.gender)
        array_food.append(self.fav_food)
        
        if self.gender == 'female':
            Animal.Numb_female +=1     
        else:
            Animal.Numb_male +=1
        return

             
class Rabbit(Animal):
    DIET = [
        'basil',
        'cabbages',
        'carrots',
        'clover',
        'cupcakes',
        'parsley',
    ]
    def __init__(self,name, gender, fav_food):
        super().__init__(name, gender, fav_food)
        self.name= name
        self.gender = gender
        self.fav_food= fav_food
        
    def greet(self):
        return f"Hi,I'm a rabbit named {self.name}. "
    
    def likes_to_eat(self):
        
        if self.fav_food in self.DIET:
            return f"I love eating {self.fav_food}."
             

class Lion(Animal):
    DIET = [
        'antelopes',
        'buffaloes',
        'crocodiles',
        'giraffes',
        'hippos',
        'rhinos',
        'wild hogs',
        'young elephants',
        'zebras',
    ]
    
    def __init__(self,name, gender, fav_food):
            
        super().__init__(name, gender, fav_food)
        self.name= name
        self.gender = gender
        self.fav_food= fav_food
            
    def greet(self):
             return f"Roar,I'm a lion named {self.name}. "
             
    def likes_to_eat(self):
        if self.fav_food in self.DIET:
            return f"I love eating {self.fav_food}."

class Zoo():

    def __init__(self, animals):
        self._animals = animals
        return

    # TODO
    def meet_and_greet(self):
        meet_and_greet =[]
        for a in self._animals:
            meet_and_greet.append( a.greet()) 
        return meet_and_greet

    # TODO
    def list_fav_foods(self):
        fav_foods=[]
        for a in self._animals:
            fav_foods.append(a.likes_to_eat())
        return fav_foods

    # TODO
    def list_unique_fav_foods(self):
        unique_foods=[]
        for a in self._animals:
            unique_foods.append(a.fav_food)
        
        unique_foods = set(unique_foods)
        unique_foods= list(unique_foods)
        unique_foods = sorted(unique_foods) # sorted
        
        return unique_foods

    # TODO
    def count_genders(self):

        for a in self._animals:
            Numb_female =a.Numb_female
            Numb_male= a.Numb_male
        count_gender={'male':Numb_male ,'female':Numb_female}

            
        return count_gender

def main():
    class_dict = {
        'rabbit': Rabbit,
        'lion': Lion,
    }
    animals = []
    num_animals = int(input())
    for i in range(0, num_animals):
        cls, name, gender, fav_food = input().split(' ')
        # animals.append(Rabbit('a'))
        animals.append(class_dict[cls.lower()](name=name, gender=gender, fav_food=fav_food))
        #print(animals[0].name) #bugs
        
    zoo = Zoo(animals)

    print('\n'.join(zoo.meet_and_greet()))
    print(', '.join(zoo.list_fav_foods()))
    print(', '.join(zoo.list_unique_fav_foods()))
    gender = zoo.count_genders()
    print(f"male: {gender.get('male', 0)}, female: {gender.get('female', 0)}")

if __name__ == "__main__":
    main()
