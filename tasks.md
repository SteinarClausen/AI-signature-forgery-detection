# task 2
## 2a
Fra felt til matrise: Verdien i punkt (x,y) settes inn i matrisen på rad i = x, og kolonne j = y

Fra matrise til vektor: element fra matrisen på m(i,j) går til vektoren med å bruke formelen v_k, k = i*W + j

Invers mapping fra vektor til matrise: i = k//W (heltalldivisjon gir raden) j=k%W, mododulo gir kolonnen

## 2b
D(s1, s2) = sqrt(sum(n,i=1) ((fs1) - f(s2))^2)

Dette gir forskjellen i intensitet for to bilder for piksel per piksel intensitet. Med lav euclidian differanse så har en en lignende signatur. 

- Det er likevel noen drawbacks med å bruke denne metoden, den er følsom til forskyvning av signaturen, en liten forskyvning vil endre hele distansen mellom to signaturer,

- Det er også forskjell på signaturer fra samme person. med variasjon fra signatur til signatur. Det kan være forskjeller i strektykkelse og form.

- Den er også begrensende med at den kun ser på pikselverdier og ikke på streker eller andre trekk.

## 2c

### task 2 c

1. Numerisk stabilitet (Gjennomsnitt og normalisering)Valget mellom 8-bit heltall (uint8) og flyttall (float) i området $[0, 1]$ har store konsekvenser for stabiliteten i matematiske operasjoner:8-bit heltall ($\{0, \dots, 255\}$):Overflow (Overflyt): Ved beregning av gjennomsnitt må man summere pikselverdier. I et bilde med mange piksler vil summen raskt overstige 255, som er maksgrensen for en 8-bit variabel. Dette fører til feilaktige "rulling" av verdier (f.eks. at $255 + 1 = 0$).Presisjonstap: Divisjon med heltall fører til avrundingsfeil. Hvis du deler en pikselverdi på et tall for å normalisere, vil resultatet ofte bli forkastet eller rundet ned til nærmeste heltall, noe som fjerner subtile nyanseforskjeller i signaturen.Flyttall ($[0, 1]$):Høy presisjon: Flyttall (som float32 eller float64) tillater desimaler, noe som er kritisk for operasjoner som normalisering til null gjennomsnitt og enhetsvarians.Stabilitet: Man unngår overflow-problemer ved store summeringer, og matematiske transformasjoner (som rotasjon eller konvolusjon) forblir nøyaktige uten at informasjon går tapt i avrunding.2. Tolkning av avstander og indreproduktHvordan vi ser på likheten mellom to signaturbilder endres basert på skalaen:Euklidsk avstand ($||I_1 - I_2||_2$):I en uint8-representasjon kan avstanden mellom to piksler være så stor som 255. Over et helt bilde (vektor i $\mathbb{R}^{HW}$) vil den totale avstanden bli et enormt tall som er vanskelig å tolke intuitivt.I en $[0, 1]$-representasjon er den maksimale avstanden per piksel 1. Dette gjør det lettere å sammenligne avstander på tvers av ulike datasett, da verdiene er uavhengige av den spesifikke bit-dybden til skanneren.Indreprodukt ($I_1 \cdot I_2$):Indreproduktet brukes ofte for å måle korrelasjon eller "overlapp" mellom to bilder.Ved bruk av verdier mellom 0 og 1 fungerer indreproduktet som et mål på hvor mye "blekk" som overlapper i de to signatur-vektorene. Hvis begge bildene er normalisert til å ha enhetslengde, vil indreproduktet direkte tilsvare cosinus-likhet, som er et standardmål for likhet i maskinlæring.Oppsummert: Flyttall i området $[0, 1]$ er nesten alltid å foretrekke for numeriske beregninger fordi det bevarer detaljer under transformasjoner og gjør matematiske likhetsmål (som avstander) mer konsistente og sammenlignbare.


