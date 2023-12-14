import torch
from torch.utils.data import Dataset, random_split
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM

assert torch.cuda.is_available()
device = torch.device("cuda")
print("Using device:", device)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

tokenizer.pad_token = tokenizer.eos_token

finetune_data = {
    "data": [
        {
            "clue": "american backed and abandoned",
            "definition": "nation",
            "length": 5,
            "steps": [
                "america backed and abandoned",
                "us backed and abandoned",
                "su and abandoned",
                "su and",
                "sudan"
            ]
        },
        {
            "clue": "a bit of caution at terms",
            "definition": "for talk",
            "length": 6,
            "steps": [
                "a bit of caution at terms",
                "a bit of cautionatterms",
                "natter"
            ]
        },
        {
            "clue": "one two bottles",
            "definition": "gas",
            "length": 4,
            "steps": [
                "one two bottles",
                "oneone bottles",
                "neon"
            ]
        },
        {
            "clue": "article about delays",
            "definition": "examined",
            "length": 8,
            "steps": [
                "article about delays",
                "an about delays",
                "an alysed",
                "analysed"
            ]
        },
        {
            "clue": "initially misleading poor tom with lears",
            "definition": "confused state",
            "length": 9,
            "steps": [
                "initially misleading poor tom with lears",
                "m poor tom with lears",
                "m poor tomlears",
                "maelstrom"
            ]
        },
        {
            "clue": "we object over having fish with fruit for",
            "definition": "sweet",
            "length": 9,
            "steps": [
                "we object over having fish with fruit for",
                "us over having fish with fruit for",
                "su having fish with fruit for",
                "su having gar with fruit for",
                "su having gar with fruit",
                "su having gar with plum",
                "sugar with plum",
                "sugarplum"
            ]
        },
        {
            "clue": "montys first thoughts about a flower",
            "definition": "show on tv",
            "length": 10,
            "steps": [
                "montys first thoughts about a flower",
                "m thoughts about a flower",
                "m mind about a flower",
                "m mind about aster",
                "mmind about aster",
                "mastermind"
            ]
        },
        {
            "clue": "tendency to grab before taking a breather on the way up",
            "definition": "aggressive",
            "length": 11,
            "steps": [
                "tendency to grab before taking a breather on the way up",
                "bent to grab before taking a breather on the way up",
                "bent to grab ere taking a breather on the way up",
                "bent to grab ere taking gill on the way up",
                "bent to grab eregill on the way up",
                "bent to grab lligere",
                "belligerent"
            ]
        },
        {
            "clue": "film oddly ignored pursuing charge admitted by london area leader of hells",
            "definition": "angels",
            "length": 8,
            "steps": [
                "film oddly ignored pursuing charge admitted by london area leader of hells",
                "im pursuing charge admitted by london area leader of hells",
                "im pursuing rap admitted by london area leader of hells",
                "im pursuing rap admitted by se leader of hells",
                "im pursuing rap admitted by se h",
                "im pursuing rap admitted by seh",
                "im pursuing seraph",
                "seraphim"
            ]
        },
        {
            "clue": "writer includes an afterthought on current",
            "definition": "digestive aid",
            "length": 6,
            "steps": [
                "writer includes an afterthought on current",
                "pen includes an afterthought on current",
                "pen includes ps on current",
                "pen includes ps on i",
                "pen includes psi",
                "pepsin"
            ]
        },
        {
            "clue": "place to go to suppress tension before start of the",
            "definition": "game",
            "length": 5,
            "steps": [
                "place to go to suppress tension before start of the",
                "loo to suppress tension before start of the",
                "loo to suppress t before start of the",
                "loo to suppress t before t",
                "loo to suppress tt",
                "lotto"
            ]
        },
        {
            "clue": "was able to dismiss unionist as",
            "definition": "indifferent",
            "length": 4,
            "steps": [
                "was able to dismiss unionist as",
                "could dismiss unionist as",
                "could dismiss unionist",
                "could dismiss u",
                "cold"
            ]
        },
        {
            "clue": "of omens start worrying",
            "definition": "a variety",
            "length": 10,
            "steps": [
                "of omens start worrying",
                "omens start worrying",
                "omensstart worrying",
                "assortment"
            ]
        },
        {
            "clue": "loves fashionable hiding nothing to",
            "definition": "wear",
            "length": 7,
            "steps": [
                "loves fashionable hiding nothing to",
                "loves fashionable hiding nothing",
                "eros fashionable hiding nothing",
                "eros in hiding nothing",
                "eros in hiding o",
                "erosin hiding o",
                "erosion"
            ]
        },
        {
            "clue": "against topless skiing outrageous",
            "definition": "scandavian",
            "length": 6,
            "steps": [
                "against topless skiing outrageous",
                "v topless skiing outrageous",
                "v topless skiing outrageous",
                "v kiing outrageous",
                "vkiing outrageous",
                "viking"
            ]
        },
        {
            "clue": "has trouble with green movement",
            "definition": "envoy",
            "length": 9,
            "steps": [
                "has trouble with green movement",
                "trouble with green movement",
                "mess with green movement",
                "messgreen movement",
                "messenger"
            ]
        },
        {
            "clue": "goas very annoyed with",
            "definition": "travelers",
            "length": 8,
            "steps": [
                "goas very annoyed with",
                "goas very annoyed",
                "goasvery annoyed",
                "voyagers"
            ]
        },
        {
            "clue": "of temperature in great balls of fire",
            "definition": "origins",
            "length": 6,
            "steps": [
                "of temperature in great balls of fire",
                "t in great balls of fire",
                "t in stars",
                "starts"
            ]
        },
        {
            "clue": "is uplifting when astride pig",
            "definition": "japanese game",
            "length": 5,
            "steps": [
                "is uplifting when astride pig",
                "si when astride pig",
                "si when astride hog",
                "si astride hog",
                "shogi"
            ]
        },
        {
            "clue": "shabby coat try new",
            "definition": "shape",
            "length": 7,
            "steps": [
                "shabby coat try new",
                "octa try new",
                "octa go new",
                "octa go n",
                "octagon"
            ]
        },
        {
            "clue": "company board retains maiden over her elevated",
            "definition": "understanding",
            "length": 13,
            "steps": [
                "company board retains maiden over her elevated",
                "co board retains maiden over her elevated",
                "co pension retains maiden over her elevated",
                "co pension retains m over her elevated",
                "co pension retains m reh elevated",
                "copension retains m reh elevated",
                "compension reh elevated",
                "comprehension"
            ]
        },
        {
            "clue": "in spain or otherwise",
            "definition": "singers",
            "length": 9,
            "steps": [
                "in spain or otherwise",
                "inspainor otherwise",
                "sopranini"
            ]
        },
        {
            "clue": "flower areas dry turning to",
            "definition": "stone",
            "length": 7,
            "steps": [
                "flower areas dry turning to",
                "rose areas dry turning to",
                "rose a teetotal turning to",
                "rose a tt turning to",
                "roseatt turning to",
                "rosetta"
            ]
        },
        {
            "clue": "is better land contracts",
            "definition": "money",
            "length": 7,
            "steps": [
                "is better land contracts",
                "better land contracts",
                "cap land contracts",
                "cap italy contracts",
                "cap ital",
                "capital"
            ]
        },
        {
            "clue": "carpet powers removed",
            "definition": "mark",
            "length": 5,
            "steps": [
                "carpet powers removed",
                "carpet p removed",
                "caret"
            ]
        },
        {
            "clue": "naughty individual goes in after energy is",
            "definition": "discharged",
            "length": 10,
            "steps": [
                "naughty individual goes in after energy is",
                "naughty individual goes in after energy",
                "naughty one goes in after energy",
                "naughty one goes in after e",
                "x rated one goes in after e",
                "e x one rated",
                "exonerated"
            ]
        },
        {
            "clue": "from aldi tesco ran badly",
            "definition": "assertions",
            "length": 12,
            "steps": [
                "from aldi tesco ran badly",
                "aldi tesco ran badly",
                "alditascoran badly",
                "declarations"
            ]
        },
        {
            "clue": "vehicle needing day to",
            "definition": "book",
            "length": 4,
            "steps": [
                "vehicle needing day to",
                "car needing day to",
                "car needing day",
                "car needing d",
                "card"
            ]
        },
        {
            "clue": "through short treatment",
            "definition": "suffer",
            "length": 5,
            "steps": [
                "through short treatment",
                "in short treatment",
                "in short cure",
                "in cur",
                "incur"
            ]
        },
        {
            "clue": "matter for reflection sit and",
            "definition": "think",
            "length": 7,
            "steps": [
                "matter for reflection sit and",
                "pus for reflection sit and",
                "sup sit and",
                "sup sit",
                "sup pose",
                "suppose"
            ]
        },
        {
            "clue": "im laughing with tales for",
            "definition": "suckers",
            "length": 7,
            "steps": [
                "im laughing with tales for",
                "lol with tales for",
                "lol with tales",
                "lol with lies",
                "lollies"
            ]
        },
        {
            "clue": "with stick up on deserted exmoor",
            "definition": "dog",
            "length": 6,
            "steps": [
                "with stick up on deserted exmoor",
                "stick up on deserted exmoor",
                "cock on deserted exmoor",
                "cock on er",
                "cocker"
            ]
        },
        {
            "clue": "kitty eating a bishops",
            "definition": "bird",
            "length": 6,
            "steps": [
                "kitty eating a bishops",
                "pot eating a bishops",
                "pot eating a right reverend",
                "pot eating a rr",
                "pot eating arr",
                "parrot"
            ]
        },
        {
            "clue": "can run inside to get american",
            "definition": "grub",
            "length": 5,
            "steps": [
                "can run inside to get american",
                "lav run inside to get american",
                "lav r inside to get american",
                "larv to get american",
                "larv to get a",
                "larva"
            ]
        },
        {
            "clue": "bottom of toddler its alarming",
            "definition": "smell",
            "length": 4,
            "steps": [
                "bottom of toddler its alarming",
                "r its alarming",
                "r eek",
                "reek"
            ]
        },
        {
            "clue": "green filling in two holes in order",
            "definition": "went round",
            "length": 8,
            "steps": [
                "green filling in two holes in order",
                "vert filling in two holes in order",
                "vert filling in o o in order",
                "vert filling in o o ok",
                "overto ok",
                "overtook"
            ]
        },
        {
            "clue": "rich celebrity",
            "definition": "who believes in destiny",
            "length": 8,
            "steps": [
                "rich celebrity",
                "fat celebrity",
                "fat a list",
                "fatalist"
            ]
        },
        {
            "clue": "gets up on time not good getting",
            "definition": "rest",
            "length": 10,
            "steps": [
                "gets up on time not good getting",
                "stands on time not good getting",
                "stands on t not good getting",
                "stands on t ill getting",
                "stands on t ill",
                "standstill"
            ]
        },
        {
            "clue": "a bad copper enters section",
            "definition": "they reckon",
            "length": 8,
            "steps": [
                "a bad copper enters section",
                "a bad cu enters section",
                "a bad cu enters s",
                "a base cu enters s",
                "abase cu enters s",
                "abacus es",
                "abacuses"
            ]
        },
        {
            "clue": "shop at the back make love with rest when light goes on",
            "definition": "place for knocking",
            "length": 8,
            "steps": [
                "shop at the back make love with rest when light goes on",
                "p make love with rest when light goes on",
                "p do o with rest when light goes on",
                "p do o with rste goes on",
                "p doorste goes on",
                "doorstep"
            ]
        },
        {
            "clue": "with desire to get laid regularly outside",
            "definition": "fruity thing",
            "length": 6,
            "steps": [
                "with desire to get laid regularly outside",
                "with itch to get laid regularly outside",
                "with itch to get li outside",
                "itch to get li outside",
                "itch li outside",
                "litchi"
            ]
        },
        {
            "clue": "upset sergeant major goes off and",
            "definition": "attacks",
            "length": 6,
            "steps": [
                "upset sergeant major goes off and",
                "upset sm goes off and",
                "upset sm rots",
                "upset smrots",
                "storms"
            ]
        },
        {
            "clue": "part for choral society",
            "definition": "further",
            "length": 4,
            "steps": [
                "part for choral society",
                "part for choralsociety",
                "part choralsociety",
                "also"
            ]
        },
        {
            "clue": "installation of lift in palladium",
            "definition": "commended",
            "length": 6,
            "steps": [
                "installation of lift in palladium",
                "installation of lift in pd",
                "installation of raise in pd",
                "praised"
            ]
        },
        {
            "clue": "writer on christian discipline",
            "definition": "mature",
            "length": 5,
            "steps": [
                "writer on christian discipline",
                "writer on religious instruction",
                "pen on religious instruction",
                "pen on ri",
                "ripen"
            ]
        },
        {
            "clue": "four inch boxes for",
            "definition": "bracelets",
            "length": 9,
            "steps": [
                "four inch boxes for",
                "hand boxes for",
                "hand boxes",
                "hand cuffs",
                "handcuffs"
            ]
        },
        {
            "clue": "problems unloading weight on large",
            "definition": "trucks",
            "length": 7,
            "steps": [
                "problems unloading weight on large",
                "worries unloading weight on large",
                "worries unloading w on large",
                "worries unloading w on l",
                "orries on l",
                "lorries"
            ]
        },
        {
            "clue": "noted midday excursionist returns",
            "definition": "confounded",
            "length": 6,
            "steps": [
                "noted midday excursionist returns",
                "midday excursionist returns",
                "mad dog returns",
                "maddog returns",
                "goddam"
            ]
        },
        {
            "clue": "retreating pets must keep it",
            "definition": "motionless",
            "length": 6,
            "steps": [
                "retreating pets must keep it",
                "retreating cats must keep it",
                "retreating citats",
                "static"
            ]
        },
        {
            "clue": "shown by couple turning and smothering last of fire",
            "definition": "guts",
            "length": 7,
            "steps": [
                "shown by couple turning and smothering last of fire",
                "shown by duo turning and smothering e",
                "shown by duo turning aned",
                "shown by duo dena",
                "duo dena",
                "duodena"
            ]
        },
        {
            "clue": "always stopped by lake",
            "definition": "swimmer using river",
            "length": 5,
            "steps": [
                "always stopped by lake",
                "ever stopped by lake",
                "ever stopped by l",
                "elver"
            ]
        },
        {
            "clue": "among clues i do dreadfully one is",
            "definition": "to be savored",
            "length": 9,
            "steps": [
                "among clues i do dreadfully one is",
                "among clues i do dreadfully i",
                "among clues i do dreadfully i",
                "among cluesido dreadfully i",
                "among delicous i",
                "delicious"
            ]
        },
        {
            "clue": "knees regulalry scrubbed in spring",
            "definition": "most dirty",
            "length": 7,
            "steps": [
                "knees regulalry scrubbed in spring",
                "kes in spring",
                "kes in dart",
                "darkest"
            ]
        },
        {
            "clue": "cut and run before church",
            "definition": "robs",
            "length": 6,
            "steps": [
                "cut and run before church",
                "flee before church",
                "flee before church of england",
                "flee before ce",
                "fleece"
            ]
        },
        {
            "clue": "alarm when engineers put in",
            "definition": "anything worn",
            "length": 7,
            "steps": [
                "alarm when engineers put in",
                "appal when engineers put in",
                "appal when royal engineers put in",
                "appal when re put in",
                "appal re put in",
                "apparel"
            ]
        },
        {
            "clue": "take tea perhaps with your",
            "definition": "sweetener",
            "length": 5,
            "steps": [
                "take tea perhaps with your",
                "sup with your",
                "sup with yr",
                "syrup"
            ]
        },
        {
            "clue": "master receiving kiss when artist finds",
            "definition": "flower",
            "length": 9,
            "steps": [
                "master receiving kiss when artist finds",
                "sage receiving kiss when artist finds",
                "sage receiving x when artist finds",
                "sage receiving x if artist finds",
                "sage receiving x if ra finds",
                "sage receiving x if ra",
                "sage receiving xifra",
                "saxifrage"
            ]
        },
        {
            "clue": "three cardinals with boring clothing",
            "definition": "far from it",
            "length": 6,
            "steps": [
                "three cardinals with boring clothing",
                "east south south with boring clothing",
                "east south south with dry clothing",
                "ess with dry clothing",
                "ess with dry",
                "dressy"
            ]
        },
        {
            "clue": "irreverent drunk docking tail of deacon",
            "definition": "dog",
            "length": 9,
            "steps": [
                "irreverent drunk docking tail of deacon",
                "irreverent drunk docking n",
                "irreveret drunk",
                "retriever"
            ]
        },
        {
            "clue": "in paris cold shivering",
            "definition": "afflicted",
            "length": 9,
            "steps": [
                "paris cold shivering",
                "pariscold shivering",
                "dropsical"
            ]
        },
        {
            "clue": "language curbed by court",
            "definition": "wwii veteran",
            "length": 7,
            "steps": [
                "language curbed by court",
                "language curbed by ct",
                "hindi curbed by ct",
                "chindit"
            ]
        },
        {
            "clue": "face punched by officer rising",
            "definition": "skilfully",
            "length": 0,
            "steps": [
                "face punched by officer rising",
                "defy punched by officer rising",
                "defy punched by lt rising",
                "defy punched by tl",
                "deftly"
            ]
        },
        {
            "clue": "playing sitar with extremely trite accompaniment",
            "definition": "performer",
            "length": 7,
            "steps": [
                "playing sitar with extremely trite accompaniment",
                "playing sitar with extremely trite",
                "playing sitar with extremely te",
                "playing sitar with te",
                "playing sitarte",
                "artiste"
            ]
        },
        {
            "clue": "on boundary",
            "definition": "superstar",
            "length": 6,
            "steps": [
                "on boundary",
                "leg boundary",
                "leg end",
                "legend"
            ]
        },
        {
            "clue": "lunch or deliver sandwiches",
            "definition": "pack",
            "length": 5,
            "steps": [
                "lunch or deliver sandwiches",
                "lunchordeliver sandwiches",
                "horde"
            ]
        },
        {
            "clue": "say including small ducks on the rise",
            "definition": "waterfowl",
            "length": 5,
            "steps": [
                "say including small ducks on the rise",
                "eg including small ducks on the rise",
                "eg including s ducks on the rise",
                "eg including s oo on the rise",
                "eg including soo on the rise",
                "esoog on the rise",
                "goose"
            ]
        },
        {
            "clue": "in job a man avoiding extremes",
            "definition": "nobelist",
            "length": 5,
            "steps": [
                "in job a man avoiding extremes",
                "job a man avoiding extremes",
                "jobaman avoiding extremes",
                "obama"
            ]
        },
        {
            "clue": "variety of rose ahead of time in bad weather",
            "definition": "plant",
            "length": 9,
            "steps": [
                "variety of rose ahead of time in bad weather",
                "orse ahead of time in bad weather",
                "orse ahead of t in bad weather",
                "orse ahead of t in hail",
                "orset in hail",
                "horsetail"
            ]
        },
        {
            "clue": "couple holding hands consuming punch or",
            "definition": "wine not eighteen",
            "length": 5,
            "steps": [
                "couple holding hands consuming hit or",
                "west east consuming hit or",
                "west east consuming hit",
                "we consuming hit",
                "white"
            ]
        },
        {
            "clue": "old hat on entering",
            "definition": "given",
            "length": 7,
            "steps": [
                "old hat on entering",
                "dated on entering",
                "donated"
            ]
        },
        {
            "clue": "part of florida house",
            "definition": "home to some americans",
            "length": 5,
            "steps": [
                "part of florida house",
                "part of floridahouse",
                "idaho"
            ]
        },
        {
            "clue": "contrived to share",
            "definition": "whats within it we hear",
            "length": 7,
            "steps": [
                "contrived to share",
                "contrived toshare",
                "earshot"
            ]
        },
        {
            "clue": "and architect in decorative fabric",
            "definition": "writer",
            "length": 8,
            "steps": [
                "and architect in decorative fabric",
                "architect in decorative fabric",
                "wren in decorative fabric",
                "wren in lace",
                "lawrence"
            ]
        },
        {
            "clue": "i badmouth",
            "definition": "irishman for instance",
            "length": 8,
            "steps": [
                "i badmouth",
                "i slander",
                "islander"
            ]
        },
        {
            "clue": "utter expression of pain",
            "definition": "like coward",
            "length": 6,
            "steps": [
                "utter expression of pain",
                "yell expression of pain",
                "yell ow",
                "yellow"
            ]
        },
        {
            "clue": "i like nothing about new",
            "definition": "source of dye",
            "length": 6,
            "steps": [
                "i like nothing about new",
                "i dig nothing about new",
                "i dig o about new",
                "i dig o about n",
                "idigo about n",
                "indigo"
            ]
        },
        {
            "clue": "recommend cast across river for",
            "definition": "fish",
            "length": 5,
            "steps": [
                "recommend cast across river for",
                "tout cast across river for",
                "tout cast across r for",
                "tout cast across r",
                "trout"
            ]
        },
        {
            "clue": "start of dance lesson say",
            "definition": "not looking forward to",
            "length": 8,
            "steps": [
                "start of dance lesson say",
                "d lesson say",
                "d reading",
                "dreading"
            ]
        },
        {
            "clue": "area rented in the south for",
            "definition": "people having events",
            "length": 8,
            "steps": [
                "area rented in the south for",
                "a rented in the south for",
                "a let in the south for",
                "a let in the south",
                "a let in the s",
                "a thlete s",
                "athletes"
            ]
        },
        {
            "clue": "girl is very strong name withheld",
            "definition": "flower",
            "length": 6,
            "steps": [
                "girl is very strong name withheld",
                "girl is violent name withheld",
                "is violent name withheld",
                "violent name withheld",
                "violent n withheld",
                "violet"
            ]
        },
        {
            "clue": "from african port for example sent north",
            "definition": "fruit",
            "length": 6,
            "steps": [
                "from african port for example sent north",
                "african port for example sent north",
                "oran for example sent north",
                "oran eg sent north",
                "oran ge",
                "orange"
            ]
        },
        {
            "clue": "killer having stolen a ring",
            "definition": "waste",
            "length": 5,
            "steps": [
                "killer having stolen a ring",
                "gun having stolen a ring",
                "gun having stolen a o",
                "gun having stolen ao",
                "guano"
            ]
        },
        {
            "clue": "back of mining enginner nearer to",
            "definition": "dynamo",
            "length": 9,
            "steps": [
                "back of mining enginner nearer to",
                "g engineer nearer to",
                "g engineer nearerto",
                "g enerator",
                "generator"
            ]
        },
        {
            "clue": "a king boarding plane thats",
            "definition": "less reliable",
            "length": 7,
            "steps": [
                "a king boarding plane thats",
                "a k boarding plane thats",
                "a k boarding plane",
                "a k boarding flier",
                "ak boarding flier",
                "flakier"
            ]
        },
        {
            "clue": "left with less information originally after amendment",
            "definition": "ordinal",
            "length": 7,
            "steps": [
                "left with less information originally after amendment",
                "left with less i after amendment",
                "left wth after amendment",
                "leftwth after amendment",
                "twelfth"
            ]
        },
        {
            "clue": "as was the queen of biblical kingdom good for",
            "definition": "an affair",
            "length": 7,
            "steps": [
                "as was the queen of biblical kingdom good for",
                "sheban good for",
                "sheban good",
                "sheban g",
                "shebang"
            ]
        },
        {
            "clue": "figure going into school after strike",
            "definition": "taken off",
            "length": 9,
            "steps": [
                "figure going into school after strike",
                "figure going into school after lam",
                "one going into school after lam",
                "one going into pod after lam",
                "pooned after lam",
                "lampooned"
            ]
        },
        {
            "clue": "iodine in lead",
            "definition": "instrument",
            "length": 5,
            "steps": [
                "iodine in lead",
                "i in lead",
                "i in star",
                "sitar"
            ]
        },
        {
            "clue": "small tots primarily on wheels in",
            "definition": "these",
            "length": 9,
            "steps": [
                "small tots primarily on wheels in",
                "st on wheels in",
                "st on wheels",
                "st on rollers",
                "strollers"
            ]
        },
        {
            "clue": "has been procured by joe",
            "definition": "some sheep",
            "length": 5,
            "steps": [
                "has been procured by joe",
                "has been procured by gi",
                "has been got by gi",
                "been got by gi",
                "got by gi",
                "gigot"
            ]
        },
        {
            "clue": "doctor aided locum",
            "definition": "as a system of counting",
            "length": 10,
            "steps": [
                "doctor aided locum",
                "doctor aidedlocum",
                "duodecimal"
            ]
        },
        {
            "clue": "in delivery truck",
            "definition": "covering for wheels",
            "length": 4,
            "steps": [
                "in delivery truck",
                "in deliverytruck",
                "eryt",
                "tyre"
            ]
        },
        {
            "clue": "house heres so dilapidated",
            "definition": "is that lucky",
            "length": 9,
            "steps": [
                "house heres so dilapidated",
                "ho heres so dilapidated",
                "ho heresso dilapidated",
                "ho rseshoe",
                "horseshoe"
            ]
        },
        {
            "clue": "refrain from cutting half baked loaves",
            "definition": "somewhere in eastern europe",
            "length": 10,
            "steps": [
                "refrain from cutting half baked loaves",
                "stop cutting half baked loaves",
                "stop cutting sevaol",
                "sevastopol"
            ]
        },
        {
            "clue": "angle courage",
            "definition": "round food",
            "length": 9,
            "steps": [
                "angle courage",
                "fish courage",
                "fish balls",
                "fishballs"
            ]
        },
        {
            "clue": "suspicion initially in patient poking not entirely cooked",
            "definition": "dish",
            "length": 9,
            "steps": [
                "suspicion initially in patient poking not entirely cooked",
                "s in patient poking not entirely cooked",
                "s in case poking not entirely cooked",
                "s in case poking not entirely fried",
                "s in case poking frie",
                "cassee poking frie",
                "fricassee"
            ]
        },
        {
            "clue": "short swirl in cleaner",
            "definition": "dairy produce",
            "length": 7,
            "steps": [
                "short eddy in cleaner",
                "edd in cleaner",
                "edd in char",
                "cheddar"
            ]
        },
        {
            "clue": "bottom covered in wake",
            "definition": "flasher and streaker",
            "length": 5,
            "steps": [
                "bottom covered in wake",
                "bottom covered come to",
                "bottom covered cometo",
                "comet"
            ]
        },
        {
            "clue": "constant love our lot",
            "definition": "devoted to god",
            "length": 5,
            "steps": [
                "constant love our lot",
                "pi love our lot",
                "pi o our lot",
                "pi o us",
                "pious"
            ]
        },
        {
            "clue": "defeat faced in a bad way",
            "definition": "as is a less stimulating cup",
            "length": 13,
            "steps": [
                "defeat faced in a bad way",
                "defeatfacedin a bad way",
                "decaffintaed"
            ]
        },
        {
            "clue": "at first actor needing new identity emulates",
            "definition": "orphan in musical theater",
            "length": 5,
            "steps": [
                "at first actor needing new identity emulates",
                "annie"
            ]
        },
        {
            "clue": "with tips of rich aqua yellow black",
            "definition": "bird",
            "length": 4,
            "steps": [
                "with tips of rich aqua yellow black",
                "hawk"
            ]
        },        
        {
            "clue": "is cowardly about to fly away",
            "definition": "bird",
            "length": 5,
            "steps": [
		        "is cowardly about to fly away",
                "cowardly about to fly away",
		        "craven about to fly away",
		        "raven"
            ]
        },
        {
            "clue": "a bit of godawful back",
            "definition": "trouble",
            "length": 3,
            "steps": [
                "a bit of godawful back",
                "a bit of lufwadog",
                "ado"
            ]
        },
        {
            "clue": "left judgeable after odd losses",
            "definition": "country",
            "length": 8,
            "steps": [
                "left judgeable after odd losses",
                "port judgeable after odd losses",
                "port ugal",
                "portugal"
            ]
        },
        {
            "clue": "outlaw leader",
            "definition": "managing money",
            "length": 7,
            "steps": [
                "outlaw leader",
                "ban leader",
                "banking"
            ]
        },
        {
            "clue": "odd stuff of mr waugh is set for",
            "definition": "someone wanting women to vote",
            "length": 10,
            "steps": [
                "odd stuff of mr waugh is set for",
                "odd stuff of mr waugh is set",
                "suffragist"
            ]
        },
        {
            "clue": "to a smart set",
            "definition": "provider of social introductions",
            "length": 11,
            "steps": [
                "to a smart set",
                "toastmaster"
            ]
        },
        {
            "clue": "a long arduous journey especially one made on foot chess piece",
            "definition": "walking",
            "length": 8,
            "steps": [
                "a long arduous journey especially one made on foot chess piece",
                "trek chess piece",
                "trek king",
                "trekking"
            ]
        },
        {
            "clue": "speak about idiot making",
            "definition": "sense",
            "length": 6,
            "steps": [
                "speak about idiot making",
                "say about idiot making",
                "say about idiot",
                "say about nit",
		        "sanity"
            ]
        }
    ]
}

finetune_data = finetune_data["data"]

class CrypticDataset(Dataset):
    PROMPT = "Find the next state to solve the cryptic crossword. Do not stop unless state has the right LENGTH. DEFINITION {definition} LENGTH {length} {stop} CLUE {clue} | STEPS {steps} STATE {state}\n\nNEXTSTATE {next_state}"

    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        tokenizer.padding_side = 'right'

        self.input_ids = []
        self.attn_masks = []
        self.labels = []

        training_texts = []
        for example in self.data:
            steps = example["steps"]
            for i in range(len(steps)):
                training_text = CrypticDataset.PROMPT.format(definition=example['definition'], length=example["length"], stop=str(len(example["steps"][i])==example["length"]),
                                                             clue=example['clue'], steps=i, state=steps[i], next_state=("STOP" if i == len(example["steps"])-1 else steps[i+1])) + "<|endoftext|>" # include the end token so model knows when to stop!
                training_texts.append(training_text)
        encodings_dict = self.tokenizer(training_texts, padding=True, truncation=True)
        for i,  training_text in enumerate(training_texts):
            self.input_ids.append(torch.tensor(encodings_dict['input_ids'][i]))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask'][i]))
            prompt_and_input_length = len(tokenizer.encode(training_text.split("NEXTSTATE")[0]+"NEXTSTATE"))
            if i == 4:
                print("{}".format(tokenizer.decode(encodings_dict['input_ids'][i], skip_special_tokens=True)))
                print(encodings_dict['input_ids'][i])
                print("{}".format(tokenizer.decode(tokenizer.encode(training_text.split("NEXTSTATE")[0]+"NEXTSTATE"), skip_special_tokens=True)))
                print(tokenizer.encode(training_text.split("NEXTSTATE")[0]+"NEXTSTATE"))
                print(torch.tensor([-100] * prompt_and_input_length + encodings_dict['input_ids'][i][prompt_and_input_length:]))
            self.labels.append(torch.tensor([-100] * prompt_and_input_length + encodings_dict['input_ids'][i][prompt_and_input_length:]))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {'input_ids': self.input_ids[idx], 'attention_mask': self.attn_masks[idx], 'labels': self.labels[idx]}

train_dataset = CrypticDataset(finetune_data, tokenizer)

from transformers import AutoModelForCausalLM
gpt2_model = AutoModelForCausalLM.from_pretrained('gpt2').to(device)

training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    weight_decay=0.05,
    save_steps=100,
    fp16=True,
    report_to="none",
)

trainer = Trainer(
    model=gpt2_model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer)

trainer.train()

{
    "clue": "county sides opening",
    "definition": "game",
    "length": 5,
    "steps": [
        "county sides opening",
        "ches sides opening",
        "ches s",
        "chess",
    ]
}

test_prompt = "Find the next state to solve the following Cryptic Crossword. DEFINITION {definition} LENGTH {length} {stop} CLUE {clue} | STEPS {steps} STATE {state}\n\nNEXTSTATE".format(
    definition="scratched", length=6, stop=False, clue="crowed about side being ", steps=0, state="crowed about side being")

test_prompt = "Find the next state to solve the following Cryptic Crossword. DEFINITION {definition} LENGTH {length} CLUE {clue} | STEPS {steps} STATE {state}\n\nNEXTSTATE".format(
    definition="nation", length=5, clue="american backed and abandoned", steps=0, state="american backed and abandoned")

thing = train_dataset[0]["input_ids"].to(device).unsqueeze(0)

inputs = tokenizer(test_prompt, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
output_ids = gpt2_model.generate(inputs, max_length=50, num_return_sequences=40, do_sample=True, temperature=0.7)
# for i, sample_output in enumerate(output_ids):
#     display("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))

max_length = 100
temperature = 0.7
generated_ids = []
input_ids = tokenizer(test_prompt, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
for _ in range(max_length):
    logits = gpt2_model(input_ids).logits[:, -1, :] / temperature
    softmax_probs = torch.softmax(logits, dim=-1)
    distribution = torch.distributions.Categorical(probs=softmax_probs)
    next_token = distribution.sample()
    generated_ids.append(next_token.detach())
    input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)

generated_text = tokenizer.decode(torch.tensor(generated_ids), skip_special_tokens=True)